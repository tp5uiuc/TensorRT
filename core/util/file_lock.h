#pragma once

#include <chrono>
#include <filesystem>

namespace torch_tensorrt {
namespace core {
namespace util {

// Cross-platform RAII file lock matching py-filelock's wire protocol so the C++ and
// Python torch-TRT runtimes can safely share a runtime cache path.
//
// Backend: Unix uses BSD flock(2); Windows uses LockFileEx on byte (0,1). The byte range
// and primitive are deliberately matched to py-filelock so a Python FileLock and a C++
// FileLock on the same .lock file conflict correctly across the runtime boundary.
//
// Usage rule: each thread / call site must construct its own FileLock. flock locks live
// per open file description, so two threads sharing one FileLock instance share one OFD
// and would not actually serialize against each other.
//
// NFS caveat: flock(2) is unreliable on older NFS kernels. py-filelock has the same
// limitation; this is not a torch-TRT-specific regression.
class FileLock {
 public:
  enum class Mode { Shared, Exclusive };

  // Opens (creates if needed) the lock file. Throws std::system_error on open failure.
  explicit FileLock(std::filesystem::path lock_path);
  ~FileLock() noexcept;

  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  FileLock(FileLock&&) noexcept;
  FileLock& operator=(FileLock&&) noexcept;

  // Blocks until acquired. Throws std::system_error on hard error.
  void lock(Mode mode);

  // Returns false on contention; throws std::system_error on hard error.
  [[nodiscard]] bool try_lock(Mode mode);

  // Polls until acquired or timeout (50ms cadence). False on timeout, throws on hard
  // error. Always makes at least one acquire attempt regardless of timeout value.
  [[nodiscard]] bool try_lock_for(Mode mode, std::chrono::milliseconds timeout);

  void unlock() noexcept;
  [[nodiscard]] bool owns_lock() const noexcept;

 private:
#ifdef _WIN32
  void* handle_ = nullptr; // HANDLE; void* keeps <windows.h> out of this header.
#else
  int fd_ = -1;
#endif
  bool owned_ = false;
  std::filesystem::path path_;
};

} // namespace util
} // namespace core
} // namespace torch_tensorrt
