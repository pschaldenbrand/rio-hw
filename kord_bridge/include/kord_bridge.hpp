#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <kord/api/kord.h>
#include <kord/api/kord_control_interface.h>
#include <kord/api/kord_receive_interface.h>

namespace rio {

class KordBridge {
public:
    static constexpr int kNumJoints = 7;

    struct State {
        std::array<double, kNumJoints> joint_q{};
        std::array<double, kNumJoints> joint_qd{};
        std::array<double, kNumJoints> joint_tau{};
        std::array<double, 6>          tcp_pose{};
        uint64_t                       tick{0};           // every waitSync — divide by elapsed s → sync Hz
        uint64_t                       stream_j_sends{0}; // every moveJ sent  — divide by elapsed s → cmd Hz
        uint64_t                       stream_l_sends{0}; // every moveL sent  — divide by elapsed s → cmd Hz
        bool                           alarm{false};
        // Raw systemAlarmState() word. Layout (see kord_receive_interface.h):
        // bits 0-3 category, 4-7 context, 8-19 condition ID, 20-23 severity.
        uint32_t                       alarm_code{0};
    };

    struct DJCCmd {
        std::array<double, kNumJoints> q{};
        std::array<double, kNumJoints> qd{};
        std::array<double, kNumJoints> qdd{};
        std::array<double, kNumJoints> tau{};
    };

    // Velocity command: bridge integrates qd at RT tick rate into a position
    // reference, clamped to joint limits.  Python jitter does not affect the
    // position reference because it is computed entirely inside the RT loop.
    struct VelCmd {
        std::array<double, kNumJoints> qd{};
        // Wide defaults: the previous ±2.09 rad limits on odd axes clipped real
        // Kassow postures and yanked vel_q_ref_ on the first Vel tick → JREF span.
        // Python sets tighter bounds only when KassowArm.joint_limits is set.
        std::array<double, kNumJoints> q_min{-6.283185, -6.283185, -6.283185, -6.283185,
                                             -6.283185, -6.283185, -6.283185};
        std::array<double, kNumJoints> q_max{ 6.283185,  6.283185,  6.283185,  6.283185,
                                              6.283185,  6.283185,  6.283185};
    };

    struct MoveJCmd {
        std::array<double, kNumJoints> q{};
        kr2::kord::TrackingType        tt{kr2::kord::TT_TIME};
        double                         tt_val{2.0};
        kr2::kord::BlendType           bt{kr2::kord::BT_NONE};
        double                         bt_val{0.0};
        kr2::kord::OverlayType         ot{kr2::kord::OT_STOPPOINT};
    };

    // Streaming moveJ: continuous high-frequency waypoint streaming.
    // Uses TT_JS_TARGET_SPEED so tt_val is a max joint speed (rad/s), not a
    // time deadline — safe regardless of position offset at first command.
    // OT_VIAPOINT lets the robot blend through each waypoint without stopping.
    struct StreamJCmd {
        std::array<double, kNumJoints> q{};
        double tt_val{0.3};   // max joint speed, rad/s
        double bt_val{0.008};   // blend window, seconds (BT_TIME)
    };

    struct MoveLCmd {
        std::array<double, 6>   tcp{};
        kr2::kord::TrackingType tt{kr2::kord::TT_TIME};
        double                  tt_val{2.0};
        kr2::kord::BlendType    bt{kr2::kord::BT_NONE};
        double                  bt_val{0.0};
        kr2::kord::OverlayType  ot{kr2::kord::OT_STOPPOINT};
    };

    // Streaming moveL: Cartesian via-point streaming with RT micro-steps.
    //
    // Python publishes a goal at ~100 Hz (KassowStation default). The RT loop
    // keeps waitSync at 250 Hz and emits OT_VIAPOINT moveL on a decimated
    // schedule. Defaults match KORD's real_time_patterns.rst: every 2nd tick
    // (~125 Hz) with TT=0.016 / BT=0.008.
    // Full-rate (throttle=1, TT=0.008) matches kord_move_linear.cpp but can drop
    // the session when the link cannot sustain a command every tick.
    // Idle (at goal) stops sending; the next goal re-anchors from measured TCP.
    //
    // max_pos_speed / max_rot_speed size each micro-step so a far goal is not
    // compressed into one short TT window (that recreates torque spikes).
    struct StreamLCmd {
        std::array<double, 6>   tcp{};
        kr2::kord::TrackingType tt{kr2::kord::TT_TIME};
        double                  tt_val{0.016};
        kr2::kord::BlendType    bt{kr2::kord::BT_TIME};
        double                  bt_val{0.008};
        double                  max_pos_speed{0.15};
        double                  max_rot_speed{0.25};
    };

    explicit KordBridge(const std::string& ip, unsigned int port = 7582,
                        unsigned int session_id = 1, int rt_priority = 80);
    ~KordBridge();

    bool connect();
    void start();
    void stop();
    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    State get_state() const;
    void  set_djc_command(const DJCCmd& cmd);
    void  set_vel_command(const VelCmd& cmd);
    void  set_stream_j_command(const StreamJCmd& cmd);
    void  set_stream_j_throttle(int n);  // send moveJ every n waitSync ticks; 1 = full rate
    void  set_stream_l_command(const StreamLCmd& cmd);
    void  set_stream_l_throttle(int n);  // send moveL every n waitSync ticks; 1 = full rate
    void  queue_move_j(const MoveJCmd& cmd);
    void  queue_move_l(const MoveLCmd& cmd);
    // Clear recoverable controller latches. Call only when the RT thread is
    // stopped — uses waitSync on this thread.
    bool clear_alarm(kr2::kord::ControlInterface::EClearRequest request);
    // CLEAR_HALT + CBUN_EVENT + UNSUSPEND (same set as kord-clean-alarm --all
    // minus CONTINUE_INIT). Needed after SafetyEvent ESTOP; --cbun alone is not
    // enough, and kord-clean-alarm keeps only the last dedicated flag unless
    // --all is used.
    bool clear_recoverable_alarms();
    bool clear_cbun_event();  // CBUN_EVENT only

private:
    enum class CmdType { None, DJC, Vel, StreamJ, StreamL, MoveJ, MoveL };

    void rt_loop();

    std::string  ip_;
    unsigned int port_;
    unsigned int session_id_;
    int          rt_priority_;

    std::shared_ptr<kr2::kord::KordCore>          kord_;
    std::unique_ptr<kr2::kord::ControlInterface>  ctl_;
    std::unique_ptr<kr2::kord::ReceiverInterface> rcv_;

    std::thread       rt_thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex state_mtx_;
    State              current_state_{};

    std::mutex cmd_mtx_;
    CmdType    active_cmd_type_{CmdType::None};
    DJCCmd     pending_djc_{};
    VelCmd     pending_vel_{};
    StreamJCmd pending_stream_j_{};
    bool       stream_j_dirty_{false};
    StreamLCmd pending_stream_l_{};
    bool       stream_l_dirty_{false};
    MoveJCmd   pending_move_j_{};
    MoveLCmd   pending_move_l_{};

    // Integrated position reference for VelCmd mode — owned by the RT thread.
    std::array<double, kNumJoints> vel_q_ref_{};

    // StreamJ throttle — n_ is atomic so Python can change it without a rebuild.
    std::atomic<int> stream_j_throttle_{2};  // send moveJ every N ticks; 1 = full sync rate
    int              stream_j_counter_{0};   // RT-thread-local, no mutex needed

    // StreamL micro-step state — owned by the RT thread (goal latched under cmd_mtx_).
    std::atomic<int> stream_l_throttle_{2};  // every 2nd tick ≈ 125 Hz (real_time_patterns)
    int              stream_l_counter_{0};
    int              stream_l_ticks_since_goal_{0};  // sync ticks since last Python goal
    std::array<double, 6> stream_l_curr_{};
    std::array<double, 6> stream_l_goal_{};
    bool stream_l_have_curr_{false};
    bool stream_l_active_{false};
};

} // namespace rio
