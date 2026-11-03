#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr size_t PAGE = 4096, SHARDS = 64;
double us(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }
void require(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
struct Args {
    std::unordered_map<std::string, std::string> m;
    Args(int argc, char** argv) {
        require(argc >= 2 && (argc-2)%2 == 0, "Usage: context_bench build|run --key value ...");
        for (int i=2; i<argc; i+=2) m.emplace(argv[i], argv[i+1]);
    }
    std::string get(std::string k, std::string fallback="") const {
        auto i=m.find("--"+k); return i==m.end() ? fallback : i->second;
    }
    size_t num(std::string k, size_t fallback) const { return std::stoull(get(k, std::to_string(fallback))); }
};
struct Vectors { int d=0; std::vector<float> x; size_t n() const { return x.size()/d; } };
Vectors fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary); require(bool(f), "Cannot open "+path);
    Vectors v; int d;
    while (f.read(reinterpret_cast<char*>(&d), 4)) {
        require(d>0 && d<100000 && (!v.d || v.d==d), "Invalid fvecs dimension"); v.d=d;
        auto pos=v.x.size(); v.x.resize(pos+d);
        require(bool(f.read(reinterpret_cast<char*>(v.x.data()+pos), d*4)), "Truncated fvecs");
    }
    require(v.d && f.eof() && f.gcount()==0, "Empty or truncated fvecs"); return v;
}
struct Chunk { uint64_t offset, length; };
struct alignas(PAGE) Page { std::array<char, PAGE> bytes; };
using PagePtr = std::shared_ptr<Page>;
struct Stats { uint64_t requests=0, hits=0, joins=0, reads=0, overlap=0; double read_us=0; };
struct Pending { size_t active=0; std::promise<PagePtr> promise; std::shared_future<PagePtr> future=promise.get_future().share(); };
struct Shard {
    std::mutex mutex;
    std::unordered_map<uint64_t, std::shared_ptr<Pending>> pending;
    std::list<uint64_t> lru;
    struct Entry { PagePtr page; std::list<uint64_t>::iterator it; };
    std::unordered_map<uint64_t, Entry> cache;
};
class Reader {
    int fd; bool share; size_t capacity;
    std::array<Shard, SHARDS> shards;
public:
    Reader(std::string path, bool direct, bool shared, size_t pages): share(shared), capacity(pages) {
        fd=open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
        require(fd>=0, "Open context failed: "+std::string(strerror(errno)));
        posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
    ~Reader() { close(fd); }
    PagePtr get(uint64_t page, Stats& st) {
        auto& sh=shards[page%SHARDS]; const auto cap=capacity/SHARDS+(page%SHARDS < capacity%SHARDS);
        std::shared_ptr<Pending> pending;
        bool follower=false;
        {
            std::lock_guard lock(sh.mutex); ++st.requests;
            if (cap) {
                auto it=sh.cache.find(page);
                if (it!=sh.cache.end()) {
                    ++st.hits; sh.lru.splice(sh.lru.begin(), sh.lru, it->second.it); return it->second.page;
                }
            }
            auto it=sh.pending.find(page);
            if (it!=sh.pending.end()) { ++st.overlap; pending=it->second; follower=share; }
            else { pending=std::make_shared<Pending>(); sh.pending.emplace(page, pending); }
            if (follower) ++st.joins; else ++pending->active;
        }
        if (follower) return pending->future.get();
        PagePtr result;
        std::exception_ptr error;
        try {
            result=std::make_shared<Page>();
            auto begin=Clock::now(); ssize_t n;
            do { n=pread(fd, result->bytes.data(), PAGE, page*PAGE); } while (n<0 && errno==EINTR);
            st.read_us+=us(Clock::now()-begin); ++st.reads;
            require(n==ssize_t(PAGE), "pread failed/short read: "+std::string(strerror(errno)));
        } catch (...) { error=std::current_exception(); }
        {
            std::lock_guard lock(sh.mutex);
            if (!error && cap) {
                auto it=sh.cache.find(page);
                if (it==sh.cache.end()) {
                    sh.lru.push_front(page); sh.cache.emplace(page, Shard::Entry{result, sh.lru.begin()});
                    if (sh.cache.size()>cap) { sh.cache.erase(sh.lru.back()); sh.lru.pop_back(); }
                } else sh.lru.splice(sh.lru.begin(), sh.lru, it->second.it);
            }
            if (share) {
                if (error) pending->promise.set_exception(error); else pending->promise.set_value(result);
            }
            if (--pending->active==0) sh.pending.erase(page);
        }
        if (error) std::rethrow_exception(error);
        return result;
    }
};
struct Sample { double total=0, ann=0, context=0; uint64_t digest=0; };
double percentile(std::vector<double> x, double q) {
    std::sort(x.begin(), x.end()); return x[std::min(x.size()-1, size_t(std::ceil(x.size()*q)-1))];
}
double cpu_seconds() { rusage r{}; getrusage(RUSAGE_SELF,&r); return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6; }
uint64_t read_bytes() {
    std::ifstream f("/proc/self/io"); std::string k; uint64_t v;
    while(f>>k>>v) if(k=="read_bytes:") return v;
    return 0;
}
struct Disk { uint64_t reads=0, merged=0, sectors=0, read_ms=0; bool valid=false; };
Disk disk_stats(const std::string& name) {
    Disk d;
    if(name.empty()) return d;
    require(name.find('/')==std::string::npos, "Device must be a kernel name, e.g. nvme0n1");
    std::ifstream f("/sys/class/block/"+name+"/stat");
    d.valid=bool(f>>d.reads>>d.merged>>d.sectors>>d.read_ms);
    require(d.valid,"Cannot read device counters"); return d;
}
int main(int argc, char** argv) try {
    Args a(argc,argv); omp_set_dynamic(0); omp_set_num_threads(1);
    if(std::string(argv[1])=="build") {
        auto v=fvecs(a.get("base")); faiss::IndexHNSWFlat index(v.d, a.num("m",16));
        index.hnsw.efConstruction=a.num("ef-construction",80);
        omp_set_num_threads(a.num("threads",8)); index.add(v.n(),v.x.data());
        faiss::write_index(&index,a.get("index").c_str()); return 0;
    }
    require(std::string(argv[1])=="run", "Unknown command");
    const auto mode=a.get("mode","none");
    require(mode=="none" || mode=="shared" || mode=="lru" || mode=="lru_shared", "Invalid mode");
    const bool cache=mode=="lru" || mode=="lru_shared";
    size_t concurrency=a.num("concurrency",1), topk=a.num("topk",10), warm=a.num("warmup",1024);
    require(concurrency && concurrency<=4096 && topk, "Invalid concurrency/topk");
    std::unique_ptr<faiss::Index> index(faiss::read_index(a.get("index").c_str()));
    auto hnsw=dynamic_cast<faiss::IndexHNSW*>(index.get()); require(hnsw, "Expected HNSW index");
    hnsw->hnsw.efSearch=a.num("ef-search",32);
    auto queries=fvecs(a.get("queries")); require(queries.d==index->d, "Query dimension mismatch");
    require(topk<=size_t(index->ntotal), "topk exceeds index size");
    std::vector<Chunk> chunks; Chunk chunk;
    std::ifstream mapping(a.get("mapping")); require(bool(mapping), "Cannot open chunk mapping");
    struct stat file_stat{}; require(stat(a.get("context").c_str(),&file_stat)==0, "Cannot stat context");
    require(file_stat.st_size>0 && file_stat.st_size%PAGE==0, "Context file must be padded to 4096 bytes");
    while(mapping>>chunk.offset>>chunk.length) {
        require(chunk.length && chunk.offset<=uint64_t(file_stat.st_size) && chunk.length<=uint64_t(file_stat.st_size)-chunk.offset, "Invalid chunk bounds");
        chunks.push_back(chunk);
    }
    require(mapping.eof() && chunks.size()==size_t(index->ntotal), "Mapping must have one offset,length pair per vector");
    std::vector<size_t> trace; size_t qid;
    std::ifstream tf(a.get("trace")); require(bool(tf), "Cannot open trace");
    while(tf>>qid) { require(qid<queries.n(), "Query ID out of range"); trace.push_back(qid); }
    require(tf.eof() && trace.size()>warm, "Empty, malformed, or too short trace");
    const auto io=a.get("io","direct"); require(io=="direct" || io=="buffered", "Invalid io mode");
    Reader reader(a.get("context"), io=="direct", mode=="shared" || mode=="lru_shared", cache?a.num("cache-pages",4096):0);
    auto query=[&](size_t trace_id, Stats& st, std::vector<faiss::idx_t>& ids, std::vector<float>& distances, std::vector<char>& restored) {
        Sample s; auto start=Clock::now();
        index->search(1,queries.x.data()+trace[trace_id]*queries.d,topk,distances.data(),ids.data());
        auto ann_end=Clock::now();
        uint64_t digest=0;
        for(auto id:ids) {
            require(id>=0 && size_t(id)<chunks.size(),"Invalid ANN result"); const auto c=chunks[id];
            restored.resize(c.length);
            for(uint64_t copied=0; copied<c.length;) {
                const auto offset=c.offset+copied, within=offset%PAGE, len=std::min(uint64_t(PAGE-within),c.length-copied);
                auto page=reader.get(offset/PAGE,st);
                memcpy(restored.data()+copied,page->bytes.data()+within,len); copied+=len;
            }
            // Consume reconstructed content identically in every mode; no text/tokenizer shortcut.
            uint64_t h=uint64_t(id); size_t j=0;
            for(; j+8<=restored.size(); j+=8) { uint64_t word; memcpy(&word,restored.data()+j,8); h+=word; }
            for(; j<restored.size(); ++j) h+=static_cast<unsigned char>(restored[j]);
            digest=digest*1099511628211ULL+h;
        }
        auto end=Clock::now(); s.total=us(end-start); s.ann=us(ann_end-start); s.context=us(end-ann_end); s.digest=digest; return s;
    };
    // Fixed prefix, same for all variants. Warmup is sequential and excluded from metrics.
    Stats warmstats; std::vector<faiss::idx_t> warmids(topk); std::vector<float> warmdist(topk); std::vector<char> warmbuf;
    for(size_t i=0;i<warm;++i) query(i,warmstats,warmids,warmdist,warmbuf);
    const size_t n=trace.size()-warm;
    std::vector<Sample> samples(n); std::vector<Stats> stats(concurrency);
    std::atomic<size_t> next{0}; std::atomic<bool> failed{false};
    std::mutex error_mutex; std::exception_ptr error;
    std::barrier ready(concurrency+1), start_gate(concurrency+1), done(concurrency+1);
    std::vector<std::thread> workers;
    for(size_t t=0;t<concurrency;++t) workers.emplace_back([&,t] {
        omp_set_num_threads(1);
        std::vector<faiss::idx_t> ids(topk); std::vector<float> distances(topk); std::vector<char> restored;
        ready.arrive_and_wait(); start_gate.arrive_and_wait();
        try {
            while(!failed.load(std::memory_order_relaxed)) {
                auto i=next.fetch_add(1,std::memory_order_relaxed); if(i>=n) break;
                samples[i]=query(i+warm,stats[t],ids,distances,restored);
            }
        } catch (...) { std::lock_guard lock(error_mutex); if(!error) error=std::current_exception(); failed=true; }
        done.arrive_and_wait();
    });
    ready.arrive_and_wait();
    std::vector<std::string> observed_names;
    std::vector<Disk> observed0, observed1;
    std::istringstream device_list(a.get("observe-devices"));
    for(std::string name; std::getline(device_list,name,',');) {
        observed_names.push_back(name); observed0.push_back(disk_stats(name));
    }
    auto disk0=disk_stats(a.get("device"));
    auto io0=read_bytes(); double cpu0=cpu_seconds(); auto begin=Clock::now();
    // Monitor can delimit the timed interval without including index load or warmup.
    std::cerr<<"MEASURE_START\n"<<std::flush;
    start_gate.arrive_and_wait(); done.arrive_and_wait();
    double seconds=us(Clock::now()-begin)/1e6, cpu=cpu_seconds()-cpu0; auto bytes=read_bytes()-io0;
    auto disk1=disk_stats(a.get("device"));
    for(const auto& name:observed_names) observed1.push_back(disk_stats(name));
    std::cerr<<"MEASURE_END\n"<<std::flush;
    for(auto& w:workers) w.join();
    if(error) std::rethrow_exception(error);
    Stats sum; for(auto s:stats) { sum.requests+=s.requests; sum.hits+=s.hits; sum.joins+=s.joins; sum.reads+=s.reads; sum.overlap+=s.overlap; sum.read_us+=s.read_us; }
    require(sum.requests==sum.hits+sum.joins+sum.reads,"Read accounting invariant failed");
    std::vector<double> latency; double ann=0, context=0; uint64_t digest=0;
    std::ofstream csv;
    if(!a.get("latencies").empty()) { csv.open(a.get("latencies")); require(bool(csv),"Cannot open latency output"); csv<<"trace_index,query_id,total_us,ann_us,context_us,digest\n"; }
    for(size_t i=0;i<n;++i) {
        auto s=samples[i]; latency.push_back(s.total); ann+=s.ann; context+=s.context; digest+=s.digest;
        if(csv) csv<<i+warm<<','<<trace[i+warm]<<','<<s.total<<','<<s.ann<<','<<s.context<<','<<s.digest<<'\n';
    }
    std::cout<<std::setprecision(12)<<"{\"mode\":\""<<mode<<"\",\"io\":\""<<io<<"\",\"concurrency\":"<<concurrency
      <<",\"queries\":"<<n<<",\"seconds\":"<<seconds<<",\"qps\":"<<n/seconds
      <<",\"p50_us\":"<<percentile(latency,.50)<<",\"p95_us\":"<<percentile(latency,.95)<<",\"p99_us\":"<<percentile(latency,.99)
      <<",\"ann_mean_us\":"<<ann/n<<",\"context_mean_us\":"<<context/n<<",\"cpu_cores\":"<<cpu/seconds
      <<",\"logical_pages\":"<<sum.requests<<",\"cache_hits\":"<<sum.hits<<",\"joined_reads\":"<<sum.joins
      <<",\"overlap_requests\":"<<sum.overlap<<",\"pread_calls\":"<<sum.reads<<",\"pread_iops\":"<<sum.reads/seconds
      <<",\"pread_mean_us\":"<<(sum.reads?sum.read_us/sum.reads:0)<<",\"process_read_bytes\":"<<bytes
      <<",\"digest\":\""<<digest<<"\"";
    std::cout<<",\"device_counters_available\":"<<(disk0.valid?"true":"false");
    if(disk0.valid) std::cout<<",\"device\":\""<<a.get("device")<<"\",\"device_reads\":"<<(disk1.reads-disk0.reads)
      <<",\"device_reads_merged\":"<<(disk1.merged-disk0.merged)
      <<",\"device_read_iops\":"<<(disk1.reads-disk0.reads)/seconds
      <<",\"device_read_bytes\":"<<(disk1.sectors-disk0.sectors)*512
      <<",\"device_read_await_ms\":"<<(disk1.reads>disk0.reads ? double(disk1.read_ms-disk0.read_ms)/(disk1.reads-disk0.reads):0);
    else std::cout<<",\"device\":null,\"device_reads\":null,\"device_reads_merged\":null,\"device_read_iops\":null,\"device_read_bytes\":null,\"device_read_await_ms\":null";
    std::cout<<",\"observed_devices\":{";
    for(size_t i=0;i<observed_names.size();++i) {
        const auto& before=observed0[i]; const auto& after=observed1[i];
        const auto reads=after.reads-before.reads;
        std::cout<<(i?",":"")<<"\""<<observed_names[i]<<"\":{\"reads\":"<<reads
          <<",\"read_bytes\":"<<(after.sectors-before.sectors)*512
          <<",\"read_iops\":"<<reads/seconds
          <<",\"read_await_ms\":"<<(reads?double(after.read_ms-before.read_ms)/reads:0)<<"}";
    }
    std::cout<<"}";
    std::cout<<"}\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
