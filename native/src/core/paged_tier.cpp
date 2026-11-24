// Generic 3-tier VRAM / Pinned HostRegion / NVMe mmap paging primitives.
#include "core/paged_tier.h"

#include "core/device.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ingine {

// ----------------------------------------------------------------------------------------------- MmapFile

MmapFile::MmapFile(const std::filesystem::path& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("MmapFile: open(" + path.string() + ") failed: " + std::strerror(errno));
  }
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    const int e = errno;
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("MmapFile: fstat(" + path.string() + ") failed: " + std::strerror(e));
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ == 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("MmapFile: empty file " + path.string());
  }
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    const int e = errno;
    ::close(fd_);
    fd_ = -1;
    size_ = 0;
    throw std::runtime_error("MmapFile: mmap(" + path.string() + ") failed: " + std::strerror(e));
  }
  addr_ = p;
  (void)::madvise(addr_, size_, MADV_RANDOM);
}

MmapFile::~MmapFile() { release(); }

MmapFile::MmapFile(MmapFile&& other) noexcept
    : addr_(std::exchange(other.addr_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      fd_(std::exchange(other.fd_, -1)) {}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
  if (this != &other) {
    release();
    addr_ = std::exchange(other.addr_, nullptr);
    size_ = std::exchange(other.size_, 0);
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

void MmapFile::release() noexcept {
  if (addr_) {
    (void)::madvise(addr_, size_, MADV_DONTNEED);
    ::munmap(addr_, size_);
  }
  if (fd_ >= 0) ::close(fd_);
  addr_ = nullptr;
  size_ = 0;
  fd_ = -1;
}

void MmapFile::read_slice(std::uint64_t offset, std::size_t bytes, void* dst) const {
  if (!addr_ || offset + bytes > size_) {
    throw std::out_of_range("MmapFile::read_slice: range out of bounds");
  }
  std::memcpy(dst, static_cast<const std::byte*>(addr_) + offset, bytes);
}

void MmapFile::advise_dontneed(std::uint64_t offset, std::size_t bytes) const noexcept {
  if (!addr_ || bytes == 0 || offset >= size_) return;
  const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  const std::uint64_t start = (offset / page) * page;
  const std::uint64_t end = std::min<std::uint64_t>(size_, ((offset + bytes + page - 1) / page) * page);
  if (end > start) {
    (void)::madvise(static_cast<std::byte*>(addr_) + start, static_cast<std::size_t>(end - start), MADV_DONTNEED);
  }
}

void MmapFile::advise_dontneed_all() const noexcept {
  if (addr_ && size_ > 0) (void)::madvise(addr_, size_, MADV_DONTNEED);
}

// --------------------------------------------------------------------------------------- PinnedBounceRing

PinnedBounceRing::PinnedBounceRing(int num_slots, std::size_t slot_bytes)
    : slot_bytes_((slot_bytes + 255) / 256 * 256), num_slots_(num_slots) {
  if (num_slots <= 0 || slot_bytes == 0) {
    throw std::invalid_argument("PinnedBounceRing: num_slots and slot_bytes must be positive");
  }
  buf_ = MappedHostBuffer(std::size_t(num_slots_) * slot_bytes_);
  done_.resize(std::size_t(num_slots_), nullptr);
  inflight_.assign(std::size_t(num_slots_), false);
  for (int i = 0; i < num_slots_; ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&done_[i], cudaEventDisableTiming));
  }
}

PinnedBounceRing::~PinnedBounceRing() { release(); }

PinnedBounceRing::PinnedBounceRing(PinnedBounceRing&& other) noexcept
    : buf_(std::move(other.buf_)),
      done_(std::move(other.done_)),
      inflight_(std::move(other.inflight_)),
      slot_bytes_(std::exchange(other.slot_bytes_, 0)),
      num_slots_(std::exchange(other.num_slots_, 0)),
      cursor_(std::exchange(other.cursor_, 0)) {}

PinnedBounceRing& PinnedBounceRing::operator=(PinnedBounceRing&& other) noexcept {
  if (this != &other) {
    release();
    buf_ = std::move(other.buf_);
    done_ = std::move(other.done_);
    inflight_ = std::move(other.inflight_);
    slot_bytes_ = std::exchange(other.slot_bytes_, 0);
    num_slots_ = std::exchange(other.num_slots_, 0);
    cursor_ = std::exchange(other.cursor_, 0);
  }
  return *this;
}

void PinnedBounceRing::release() noexcept {
  for (cudaEvent_t e : done_) {
    if (e) cudaEventDestroy(e);
  }
  done_.clear();
  inflight_.clear();
  num_slots_ = 0;
  slot_bytes_ = 0;
  cursor_ = 0;
}

PinnedBounceRing::Slot PinnedBounceRing::acquire() {
  const int idx = cursor_;
  cursor_ = (cursor_ + 1) % num_slots_;
  if (inflight_[idx]) {
    CUDA_CHECK(cudaEventSynchronize(done_[idx]));
    inflight_[idx] = false;
  }
  return Slot{idx, static_cast<std::uint8_t*>(buf_.data()) + std::size_t(idx) * slot_bytes_};
}

void PinnedBounceRing::mark_submitted(int index, cudaStream_t copy_stream) {
  CUDA_CHECK(cudaEventRecord(done_[index], copy_stream));
  inflight_[index] = true;
}

// -------------------------------------------------------------------------------------------- VramLruSlab

VramLruSlab::VramLruSlab(int num_slots, std::size_t slot_bytes)
    : slot_bytes_((slot_bytes + 255) / 256 * 256) {
  if (num_slots <= 0 || slot_bytes == 0) return;
  storage_ = DeviceBuffer(std::size_t(num_slots) * slot_bytes_);
  slots_.resize(std::size_t(num_slots));
  auto* base = static_cast<std::uint8_t*>(storage_.p);
  for (int i = 0; i < num_slots; ++i) {
    slots_[i].vram_ptr = base + std::size_t(i) * slot_bytes_;
  }
  index_.reserve(std::size_t(num_slots) * 2);
}

int VramLruSlab::find(std::uint64_t key) const noexcept {
  auto it = index_.find(key);
  return it == index_.end() ? -1 : it->second;
}

int VramLruSlab::touch_and_pin(std::uint64_t key, std::uint64_t pin_gen) noexcept {
  auto it = index_.find(key);
  if (it == index_.end()) return -1;
  Slot& s = slots_[it->second];
  s.last_use = ++tick_;
  s.pin_gen = pin_gen;
  return it->second;
}

VramLruSlab::AcquireResult VramLruSlab::acquire(std::uint64_t key, std::uint64_t pin_gen) {
  AcquireResult r{};
  if (slots_.empty()) throw std::runtime_error("VramLruSlab::acquire on empty slab");
  if (auto it = index_.find(key); it != index_.end()) {
    Slot& s = slots_[it->second];
    s.last_use = ++tick_;
    s.pin_gen = pin_gen;
    r.slot = it->second;
    r.vram_ptr = s.vram_ptr;
    r.hit = true;
    return r;
  }
  // Pick an empty slot first, else least-recently-used unpinned slot.
  int victim = -1;
  std::uint64_t best_use = std::numeric_limits<std::uint64_t>::max();
  for (int i = 0; i < int(slots_.size()); ++i) {
    if (slots_[i].key == kEmptyKey) {
      victim = i;
      break;
    }
    if (slots_[i].pin_gen == pin_gen) continue;
    if (slots_[i].last_use < best_use) {
      best_use = slots_[i].last_use;
      victim = i;
    }
  }
  if (victim < 0) {
    throw std::runtime_error("VramLruSlab::acquire: all " + std::to_string(slots_.size()) +
                             " slots are pinned in the same step");
  }
  Slot& s = slots_[victim];
  if (s.key != kEmptyKey) {
    r.evicted = true;
    r.evicted_key = s.key;
    index_.erase(s.key);
  }
  s.key = key;
  s.last_use = ++tick_;
  s.pin_gen = pin_gen;
  index_[key] = victim;
  r.slot = victim;
  r.vram_ptr = s.vram_ptr;
  r.hit = false;
  return r;
}

void VramLruSlab::clear() noexcept {
  index_.clear();
  for (auto& s : slots_) {
    s.key = kEmptyKey;
    s.last_use = 0;
    s.pin_gen = 0;
  }
}

// ---------------------------------------------------------------------------------------- TieredDmaStream

TieredDmaStream::TieredDmaStream() {
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
}

TieredDmaStream::~TieredDmaStream() { release(); }

TieredDmaStream::TieredDmaStream(TieredDmaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)), done_(std::exchange(other.done_, nullptr)) {}

TieredDmaStream& TieredDmaStream::operator=(TieredDmaStream&& other) noexcept {
  if (this != &other) {
    release();
    stream_ = std::exchange(other.stream_, nullptr);
    done_ = std::exchange(other.done_, nullptr);
  }
  return *this;
}

void TieredDmaStream::release() noexcept {
  if (done_) cudaEventDestroy(done_);
  if (stream_) cudaStreamDestroy(stream_);
  done_ = nullptr;
  stream_ = nullptr;
}

void TieredDmaStream::copy_h2d_async(void* dst_device, const void* src_pinned_host, std::size_t bytes) {
  CUDA_CHECK(cudaMemcpyAsync(dst_device, src_pinned_host, bytes, cudaMemcpyHostToDevice, stream_));
}

void TieredDmaStream::copy_d2h_async(void* dst_pinned_host, const void* src_device, std::size_t bytes) {
  CUDA_CHECK(cudaMemcpyAsync(dst_pinned_host, src_device, bytes, cudaMemcpyDeviceToHost, stream_));
}

void TieredDmaStream::record_and_wait_on(cudaStream_t compute_stream) {
  CUDA_CHECK(cudaEventRecord(done_, stream_));
  CUDA_CHECK(cudaStreamWaitEvent(compute_stream, done_, 0));
}

void TieredDmaStream::synchronize() const {
  if (stream_) CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---------------------------------------------------------------------------------------- process_rss_bytes

std::uint64_t process_rss_bytes() {
  std::ifstream in("/proc/self/status");
  std::string key;
  while (in >> key) {
    if (key == "VmRSS:") {
      std::uint64_t kb = 0;
      if (in >> kb) return kb * 1024ull;
      break;
    }
    std::string rest;
    std::getline(in, rest);
  }
  return 0;
}

}  // namespace ingine
