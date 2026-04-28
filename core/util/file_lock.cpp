#include "core/util/file_lock.h"

#include <chrono>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace torch_tensorrt {
namespace core {
namespace util {

namespace {

constexpr std::chrono::milliseconds kPollInterval{50};

#ifdef _WIN32

void* open_handle(const std::filesystem::path& path) {
  HANDLE h = ::CreateFileW(
      path.wstring().c_str(),
      GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE,
      nullptr,
      OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    throw std::system_error(
        static_cast<int>(::GetLastError()),
        std::system_category(),
        "FileLock: CreateFileW failed for " + path.string());
  }
  return h;
}

bool windows_lock(void* handle, FileLock::Mode mode, bool blocking) {
  DWORD flags = (mode == FileLock::Mode::Exclusive) ? LOCKFILE_EXCLUSIVE_LOCK : 0;
  if (!blocking) {
    flags |= LOCKFILE_FAIL_IMMEDIATELY;
  }
  OVERLAPPED ovl{};
  if (::LockFileEx(handle, flags, 0, 1, 0, &ovl)) {
    return true;
  }
  DWORD err = ::GetLastError();
  if (!blocking && err == ERROR_LOCK_VIOLATION) {
    return false;
  }
  throw std::system_error(static_cast<int>(err), std::system_category(), "FileLock: LockFileEx failed");
}

void windows_unlock(void* handle) noexcept {
  OVERLAPPED ovl{};
  ::UnlockFileEx(handle, 0, 1, 0, &ovl);
}

#else

int open_fd(const std::filesystem::path& path) {
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(), "FileLock: open failed for " + path.string());
  }
  return fd;
}

bool unix_lock(int fd, FileLock::Mode mode, bool blocking) {
  int operation = (mode == FileLock::Mode::Exclusive) ? LOCK_EX : LOCK_SH;
  if (!blocking) {
    operation |= LOCK_NB;
  }
  while (true) {
    if (::flock(fd, operation) == 0) {
      return true;
    }
    int err = errno;
    if (err == EINTR) {
      continue;
    }
    if (!blocking && (err == EWOULDBLOCK || err == EAGAIN)) {
      return false;
    }
    throw std::system_error(err, std::generic_category(), "FileLock: flock failed");
  }
}

void unix_unlock(int fd) noexcept {
  while (::flock(fd, LOCK_UN) == -1 && errno == EINTR) {
  }
}

#endif

} // namespace

FileLock::FileLock(std::filesystem::path lock_path) : path_(std::move(lock_path)) {
#ifdef _WIN32
  handle_ = open_handle(path_);
#else
  fd_ = open_fd(path_);
#endif
}

FileLock::~FileLock() noexcept {
  if (owned_) {
    unlock();
  }
#ifdef _WIN32
  if (handle_ != nullptr) {
    ::CloseHandle(handle_);
    handle_ = nullptr;
  }
#else
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
#endif
}

FileLock::FileLock(FileLock&& other) noexcept : owned_(other.owned_), path_(std::move(other.path_)) {
#ifdef _WIN32
  handle_ = other.handle_;
  other.handle_ = nullptr;
#else
  fd_ = other.fd_;
  other.fd_ = -1;
#endif
  other.owned_ = false;
}

FileLock& FileLock::operator=(FileLock&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (owned_) {
    unlock();
  }
#ifdef _WIN32
  if (handle_ != nullptr) {
    ::CloseHandle(handle_);
  }
  handle_ = other.handle_;
  other.handle_ = nullptr;
#else
  if (fd_ >= 0) {
    ::close(fd_);
  }
  fd_ = other.fd_;
  other.fd_ = -1;
#endif
  owned_ = other.owned_;
  other.owned_ = false;
  path_ = std::move(other.path_);
  return *this;
}

void FileLock::lock(Mode mode) {
#ifdef _WIN32
  windows_lock(handle_, mode, /*blocking=*/true);
#else
  unix_lock(fd_, mode, /*blocking=*/true);
#endif
  owned_ = true;
}

bool FileLock::try_lock(Mode mode) {
#ifdef _WIN32
  bool acquired = windows_lock(handle_, mode, /*blocking=*/false);
#else
  bool acquired = unix_lock(fd_, mode, /*blocking=*/false);
#endif
  if (acquired) {
    owned_ = true;
  }
  return acquired;
}

bool FileLock::try_lock_for(Mode mode, std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    if (try_lock(mode)) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(kPollInterval);
  }
}

void FileLock::unlock() noexcept {
  if (!owned_) {
    return;
  }
#ifdef _WIN32
  windows_unlock(handle_);
#else
  unix_unlock(fd_);
#endif
  owned_ = false;
}

bool FileLock::owns_lock() const noexcept {
  return owned_;
}

} // namespace util
} // namespace core
} // namespace torch_tensorrt
