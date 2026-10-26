/* BXI gripper mm firmware codec with the colleague's unchanged STREAM follower.
 * No PCI driver, power, homing or backend thread is linked or run.
 */
#define main unused_colleague_main
#include "gripper_backend.c"
#undef main
#define API __attribute__((visibility("default")))

typedef struct {
    motion_state motion;
    float p, v, kp, kd;
    float speed, acceleration, target_kp, target_kd, v_limit, f_limit;
} demo_context;
static pthread_mutex_t bridge_lock = PTHREAD_MUTEX_INITIALIZER;

API int demo_abi(void) { return 2; } /* Physical position: mm, velocity: mm/s, force: N. */

API int demo_pack(float p, float v, float kp, float kd, float force,
                  float v_limit, float f_limit, uint8_t data[8])
{
    if (!data || !isfinite(p) || p < 0 || p > 90 || !isfinite(v) ||
        !isfinite(kp) || kp < 0 || kp > 5 || !isfinite(kd) || kd < 0 || kd > 1 ||
        !isfinite(force) || !isfinite(v_limit) || v_limit <= 0 ||
        !isfinite(f_limit) || f_limit <= 0 || fabsf(v) > v_limit || fabsf(force) > f_limit) return -1;
    const uint32_t pi = float_to_uint(p, 0, 90, 16);
    const uint32_t vi = float_to_uint(v, -v_limit, v_limit, 12);
    const uint32_t ki = float_to_uint(kp, 0, 5, 12);
    const uint32_t di = float_to_uint(kd, 0, 1, 12);
    const uint32_t fi = force == 0 ? 0x7FF : float_to_uint(force, -f_limit, f_limit, 12);
    data[0] = pi >> 8; data[1] = pi;
    data[2] = vi >> 4; data[3] = ((vi & 15) << 4) | (ki >> 8); data[4] = ki;
    data[5] = di >> 4; data[6] = ((di & 15) << 4) | (fi >> 8); data[7] = fi;
    return 0;
}

API void *demo_new(float position_mm, float speed_mm_s, float acceleration_mm_s2,
                   float kp, float kd, float v_limit, float f_limit)
{
    if (!isfinite(position_mm) || position_mm < 0 || position_mm > 80.56f ||
        !isfinite(speed_mm_s) || speed_mm_s <= 0 || !isfinite(acceleration_mm_s2) || acceleration_mm_s2 <= 0 ||
        !isfinite(kp) || kp < 0 || kp > 5 || !isfinite(kd) || kd < 0 || kd > 1 ||
        !isfinite(v_limit) || v_limit <= 0 || speed_mm_s > v_limit || !isfinite(f_limit) || f_limit <= 0) return NULL;
    demo_context *c = calloc(1, sizeof(*c));
    if (c) {
        c->p=position_mm; c->kd=kd; c->speed=speed_mm_s; c->acceleration=acceleration_mm_s2;
        c->target_kp=kp; c->target_kd=kd; c->v_limit=v_limit; c->f_limit=f_limit;
    }
    return c;
}

API void demo_free(void *context) { free(context); }

API int demo_step(void *context, float target_mm, float dt, int follow, uint8_t data[8])
{
    if (!context || !data || !isfinite(target_mm) || target_mm < 0 || target_mm > 70 ||
        !isfinite(dt) || dt <= 0) return -1;
    demo_context *c=context;
    pthread_mutex_lock(&bridge_lock);
    g_motion=c->motion; g_command_position=c->p; g_command_velocity=c->v;
    /* update_stream_locked is unit independent. Bypass the old degree/radian UI. */
    g_motion.kind=MOTION_STREAM; g_motion.target_position=target_mm;
    g_motion.peak_velocity=c->speed; g_motion.acceleration=c->acceleration;
    dt=clampf(dt, .001f, .02f);
    if (follow) update_stream_locked(dt);
    else { g_motion.kind=MOTION_IDLE; g_command_velocity=0; }
    /* Application commands cannot exceed the confirmed 0..70 mm stroke.
     * A start above 70 mm is allowed to return continuously from its measured position. */
    const float upper=fmaxf(70.0f,c->p);
    if (g_command_position < 0) { g_command_position=0; g_command_velocity=0; }
    if (g_command_position > upper) { g_command_position=upper; g_command_velocity=0; }
    const float blend=1-expf(-dt/.02f);
    c->kp+=(c->target_kp-c->kp)*blend;
    c->kd+=(c->target_kd-c->kd)*blend;
    int result=demo_pack(g_command_position,g_command_velocity,c->kp,c->kd,0,c->v_limit,c->f_limit,data);
    c->motion=g_motion; c->p=g_command_position; c->v=g_command_velocity;
    pthread_mutex_unlock(&bridge_lock);
    return result;
}

API int demo_feedback(const uint8_t data[8], float v_limit, float f_limit, float values[5])
{
    if (!data || !values || data[0] != 1 || !isfinite(v_limit) || v_limit <= 0 ||
        !isfinite(f_limit) || f_limit <= 0) return -1;
    const uint32_t pi=(data[1]<<8)|data[2];
    const uint32_t vi=(data[3]<<4)|(data[4]>>4);
    const uint32_t fi=((data[4]&15)<<8)|data[5];
    values[0]=uint_to_float(pi,0,90,16);
    values[1]=uint_to_float(vi,-v_limit,v_limit,12);
    values[2]=uint_to_float(fi,-f_limit,f_limit,12);
    /* This profile explicitly requires AUX disabled (default NTC frame). */
    values[3]=uint_to_float(data[6],-30,150,8);
    values[4]=uint_to_float(data[7],-30,150,8);
    return 0;
}
