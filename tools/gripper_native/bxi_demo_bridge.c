/* Compile the colleague's unmodified backend; link only the functions used here.
 * No backend main/control thread, PCI driver, power, zeroing or TCP server runs.
 * Each context owns its reference. Original globals are swapped under a mutex.
 */
#define main unused_colleague_main
#include "gripper_backend.c"
#undef main
#define API __attribute__((visibility("default")))

typedef struct {
    motion_state motion;
    float p, v, kp, kd;
} demo_context;
static pthread_mutex_t bridge_lock = PTHREAD_MUTEX_INITIALIZER;

API int demo_abi(void) { return 1; }

API void *demo_new(float position)
{
    if (!isfinite(position)) return NULL;
    demo_context *c = calloc(1, sizeof(*c));
    if (c) { c->p = position; c->kd = 0.25f; }
    return c;
}

API void demo_free(void *context) { free(context); }

API int demo_step(void *context, float target_deg, float dt, int follow, uint8_t data[8])
{
    if (!context || !data || !isfinite(target_deg) || target_deg < 0.0f ||
        target_deg > 300.0f || !isfinite(dt) || dt <= 0.0f) return -1;
    demo_context *c = context;
    pthread_mutex_lock(&bridge_lock);
    g_motion = c->motion;
    g_command_position = c->p;
    g_command_velocity = c->v;
    g_command_kp = c->kp;
    g_command_kd = c->kd;
    /* Software flags only. Robot power/enable are owned by the ROS integration. */
    g_power_on = g_mode_entered = g_control_ready = 1;
    start_motion(target_deg, 1400.0f, 25.0f, 0.25f, 1);
    dt = clampf(dt, 0.001f, 0.02f); /* Original control_loop time-step bounds. */
    if (follow) update_stream_locked(dt);
    else { g_motion.kind = MOTION_IDLE; g_command_velocity = 0.0f; }
    const float blend = 1.0f - expf(-dt / 0.02f);
    g_command_kp += (g_target_kp - g_command_kp) * blend;
    g_command_kd += (g_target_kd - g_command_kd) * blend;
    pack_control(data, g_command_position, g_command_velocity, g_command_kp, g_command_kd);
    c->motion = g_motion;
    c->p = g_command_position; c->v = g_command_velocity;
    c->kp = g_command_kp; c->kd = g_command_kd;
    pthread_mutex_unlock(&bridge_lock);
    return 0;
}

API int demo_feedback(const uint8_t data[8], float values[5])
{
    if (!data || !values) return -1;
    pthread_mutex_lock(&bridge_lock);
    canfd_packet packet = {0};
    packet.bus = g_bus; packet.frame.can_id = 0x11;
    packet.frame.len = 8; packet.frame.flags = CANFD_BRS | CANFD_FDF;
    memcpy(packet.frame.data, data, 8);
    unsigned int previous = g_feedback.sequence;
    /* Decode using the original callback, without its first-ready notification. */
    g_control_ready = 1;
    pci_rx_callback(NULL, &packet);
    int accepted = g_feedback.sequence != previous;
    if (accepted) {
        values[0] = g_feedback.position; values[1] = g_feedback.velocity;
        values[2] = g_feedback.torque; values[3] = g_feedback.mos_temp;
        values[4] = g_feedback.rotor_temp;
    }
    pthread_mutex_unlock(&bridge_lock);
    return accepted ? 0 : -1;
}
