// Exercise the actual shared-library read_frame with fake ioctl and clocks.
// No device, ROS node, or robot process is opened.
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include "usb_cam/usb_cam.hpp"

static bool mocking=false;
static int64_t mono_us=100000000;
static int64_t offset_us=1792670980141125LL;
static constexpr int64_t initial_offset=1792670980141125LL;
static constexpr int64_t age_us=121000;
static unsigned int flags=V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
static int queued=0;

extern "C" int gettimeofday(timeval* tv, void*) noexcept {
  if (!mocking) return syscall(SYS_gettimeofday,tv,nullptr);
  const auto us=mono_us+offset_us;tv->tv_sec=us/1000000;tv->tv_usec=us%1000000;return 0;
}
extern "C" int clock_gettime(clockid_t id,timespec* ts) noexcept {
  if (!mocking) return syscall(SYS_clock_gettime,id,ts);
  if (id!=CLOCK_MONOTONIC) return -1;
  ts->tv_sec=mono_us/1000000;ts->tv_nsec=(mono_us%1000000)*1000;return 0;
}
extern "C" int ioctl(int, unsigned long request, ...) noexcept {
  if (!mocking) return -1; // Tests never forward device calls.
  request &= 0xffffffffUL; // The existing driver passes ioctl IDs through int.
  va_list args;va_start(args,request);void* arg=va_arg(args,void*);va_end(args);
  if (request==VIDIOC_G_FMT) return 0;
  if (request==VIDIOC_QBUF) {++queued;return 0;}
  if (request!=VIDIOC_DQBUF) return -1;
  auto* b=static_cast<v4l2_buffer*>(arg);
  b->index=0;b->bytesused=3;b->flags=flags;
  b->timestamp.tv_sec=(mono_us-age_us)/1000000;
  b->timestamp.tv_usec=(mono_us-age_us)%1000000;return 0;
}

int main(int argc,char** argv) {
  const bool baseline=argc==2 && std::strcmp(argv[1],"--expect-bug")==0;
  mocking=true;
  usb_cam::UsbCam cam; // Private members exposed only in a test-only header copy.
  char src[3]={1,2,3},dst[3]={};
  cam.m_buffers[0].start=src;cam.m_buffers[0].length=3;
  cam.m_image.data=dst;cam.m_image.size_in_bytes=3;
  cam.m_image.pixel_format=std::make_shared<usb_cam::RGB8>();
  for (int64_t step : {0LL,564615LL,-250000LL,123456LL,0LL}) {
    offset_us=initial_offset+step;mono_us+=1000000;
    cam.read_frame();const auto stamp=cam.get_image_timestamp();
    const int64_t image_us=stamp.tv_sec*1000000LL+stamp.tv_nsec/1000;
    const int64_t error=image_us-(mono_us-age_us+offset_us);
    std::cout<<"clock_step_us="<<step<<" header_error_us="<<error<<'\n';
    if (error!=(baseline?-step:0) || std::memcmp(src,dst,3)!=0) return 1;
  }
  if (queued!=5) return 2;
  // Preserve legacy behavior for unclassified buffer clocks.
  flags=0;offset_us=initial_offset+400000;cam.read_frame();
  const auto stamp=cam.get_image_timestamp();
  if (stamp.tv_sec*1000000LL+stamp.tv_nsec/1000!=mono_us-age_us+initial_offset) return 3;
  std::cout<<(baseline?"CACHED_OFFSET_BUG_REPRODUCED":"LIVE_CLOCK_CONVERSION_PASS")<<"; hardware_commands=0\n";
}
