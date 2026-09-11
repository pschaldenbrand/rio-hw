#include "kord_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <pthread.h>

using namespace kr2::kord;
using EJV = ReceiverInterface::EJointValue;

namespace rio {
namespace {

struct Quat {
    double w{1.0}, x{0.0}, y{0.0}, z{0.0};
};

Quat rotvec_to_quat(double rx, double ry, double rz)
{
    const double angle = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (angle < 1e-12) {
        return {};
    }
    const double half = 0.5 * angle;
    const double s = std::sin(half) / angle;
    return {std::cos(half), rx * s, ry * s, rz * s};
}

void quat_to_rotvec(const Quat& q, double& rx, double& ry, double& rz)
{
    Quat n = q;
    const double nrm = std::sqrt(n.w * n.w + n.x * n.x + n.y * n.y + n.z * n.z);
    if (nrm < 1e-12) {
        rx = ry = rz = 0.0;
        return;
    }
    n.w /= nrm;
    n.x /= nrm;
    n.y /= nrm;
    n.z /= nrm;
    if (n.w < 0.0) {
        n.w = -n.w;
        n.x = -n.x;
        n.y = -n.y;
        n.z = -n.z;
    }
    const double sin_half = std::sqrt(std::max(0.0, 1.0 - n.w * n.w));
    if (sin_half < 1e-12) {
        rx = ry = rz = 0.0;
        return;
    }
    const double angle = 2.0 * std::atan2(sin_half, n.w);
    const double s = angle / sin_half;
    rx = n.x * s;
    ry = n.y * s;
    rz = n.z * s;
}

Quat slerp(Quat a, Quat b, double t)
{
    double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (dot < 0.0) {
        b.w = -b.w;
        b.x = -b.x;
        b.y = -b.y;
        b.z = -b.z;
        dot = -dot;
    }
    if (dot > 0.9995) {
        Quat r{
            a.w + t * (b.w - a.w),
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z),
        };
        const double nrm = std::sqrt(r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z);
        r.w /= nrm;
        r.x /= nrm;
        r.y /= nrm;
        r.z /= nrm;
        return r;
    }
    const double theta_0 = std::acos(std::clamp(dot, -1.0, 1.0));
    const double sin_theta_0 = std::sin(theta_0);
    const double theta = theta_0 * t;
    const double s0 = std::sin(theta_0 - theta) / sin_theta_0;
    const double s1 = std::sin(theta) / sin_theta_0;
    return {
        s0 * a.w + s1 * b.w,
        s0 * a.x + s1 * b.x,
        s0 * a.y + s1 * b.y,
        s0 * a.z + s1 * b.z,
    };
}

double rot_angle(const std::array<double, 6>& a, const std::array<double, 6>& b)
{
    const Quat qa = rotvec_to_quat(a[3], a[4], a[5]);
    const Quat qb = rotvec_to_quat(b[3], b[4], b[5]);
    double dot = std::abs(qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z);
    dot = std::clamp(dot, 0.0, 1.0);
    return 2.0 * std::acos(dot);
}

double pos_dist(const std::array<double, 6>& a, const std::array<double, 6>& b)
{
    const double dx = b[0] - a[0];
    const double dy = b[1] - a[1];
    const double dz = b[2] - a[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Advance curr toward goal by at most max_pos_speed*dt translation and
// max_rot_speed*dt rotation, keeping XYZ and orientation synchronized.
bool advance_pose(std::array<double, 6>& curr, const std::array<double, 6>& goal,
                  double max_pos_speed, double max_rot_speed, double dt,
                  double pos_eps, double rot_eps)
{
    const double d_pos = pos_dist(curr, goal);
    const double d_rot = rot_angle(curr, goal);
    if (d_pos < pos_eps && d_rot < rot_eps) {
        curr = goal;
        return false;
    }

    const double step_pos = std::max(0.0, max_pos_speed) * dt;
    const double step_rot = std::max(0.0, max_rot_speed) * dt;
    double alpha = 1.0;
    if (d_pos > pos_eps) {
        alpha = std::min(alpha, step_pos / d_pos);
    }
    if (d_rot > rot_eps) {
        alpha = std::min(alpha, step_rot / d_rot);
    }
    alpha = std::clamp(alpha, 0.0, 1.0);

    std::array<double, 6> next = curr;
    for (int i = 0; i < 3; ++i) {
        next[i] = curr[i] + alpha * (goal[i] - curr[i]);
    }
    const Quat q = slerp(
        rotvec_to_quat(curr[3], curr[4], curr[5]),
        rotvec_to_quat(goal[3], goal[4], goal[5]),
        alpha);
    quat_to_rotvec(q, next[3], next[4], next[5]);
    curr = next;
    return true;
}

} // namespace

KordBridge::KordBridge(const std::string& ip, unsigned int port,
                       unsigned int session_id, int rt_priority)
    : ip_(ip), port_(port), session_id_(session_id), rt_priority_(rt_priority)
{
}

KordBridge::~KordBridge()
{
    stop();
}

bool KordBridge::connect()
{
    kord_ = std::make_shared<KordCore>(ip_, port_, session_id_, UDP_CLIENT);
    ctl_  = std::make_unique<ControlInterface>(kord_);
    rcv_  = std::make_unique<ReceiverInterface>(kord_);
    return kord_->connect();
}

void KordBridge::start()
{
    if (running_.load(std::memory_order_relaxed)) return;
    running_.store(true, std::memory_order_relaxed);
    rt_thread_ = std::thread(&KordBridge::rt_loop, this);
}

void KordBridge::stop()
{
    running_.store(false, std::memory_order_relaxed);
    if (rt_thread_.joinable()) rt_thread_.join();
    if (kord_) kord_->disconnect();
}

KordBridge::State KordBridge::get_state() const
{
    std::lock_guard<std::mutex> lock(state_mtx_);
    return current_state_;
}

void KordBridge::set_djc_command(const DJCCmd& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    pending_djc_     = cmd;
    active_cmd_type_ = CmdType::DJC;
}

void KordBridge::set_vel_command(const VelCmd& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    pending_vel_     = cmd;
    active_cmd_type_ = CmdType::Vel;
}

void KordBridge::set_stream_j_command(const StreamJCmd& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    pending_stream_j_ = cmd;
    active_cmd_type_  = CmdType::StreamJ;
    stream_j_dirty_   = true;
}

void KordBridge::set_stream_j_throttle(int n)
{
    stream_j_throttle_.store(std::max(1, n), std::memory_order_relaxed);
}

void KordBridge::set_stream_l_command(const StreamLCmd& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    pending_stream_l_ = cmd;
    active_cmd_type_  = CmdType::StreamL;
    stream_l_dirty_   = true;
}

void KordBridge::set_stream_l_throttle(int n)
{
    stream_l_throttle_.store(std::max(1, n), std::memory_order_relaxed);
}

void KordBridge::queue_move_j(const MoveJCmd& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    pending_move_j_  = cmd;
    active_cmd_type_ = CmdType::MoveJ;
}

void KordBridge::queue_move_l(const MoveLCmd& cmd)
{
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    pending_move_l_  = cmd;
    active_cmd_type_ = CmdType::MoveL;
}

bool KordBridge::clear_alarm(kr2::kord::ControlInterface::EClearRequest request)
{
    if (running_.load(std::memory_order_relaxed) || !ctl_ || !rcv_ || !kord_) {
        return false;
    }
    const int64_t token = ctl_->clearAlarmRequest(request);
    if (token == 0) {
        return false;
    }
    for (int i = 0; i < 200; ++i) {
        if (!kord_->waitSync(std::chrono::milliseconds(10), kr2::kord::F_SYNC_FULL_ROTATION)) {
            return false;
        }
        rcv_->fetchData();
        if (rcv_->getCommandStatus(token) != -1) {
            return rcv_->getCommandStatus(token) == 0;
        }
    }
    return false;
}

bool KordBridge::clear_recoverable_alarms()
{
    using EClear = kr2::kord::ControlInterface::EClearRequest;
    // Order matches kord_move_joints_autorecover / kord-clean-alarm --all.
    bool ok = clear_alarm(EClear::CLEAR_HALT);
    ok = clear_alarm(EClear::CBUN_EVENT) || ok;
    ok = clear_alarm(EClear::UNSUSPEND) || ok;
    return ok;
}

bool KordBridge::clear_cbun_event()
{
    using EClear = kr2::kord::ControlInterface::EClearRequest;
    return clear_alarm(EClear::CBUN_EVENT);
}

void KordBridge::rt_loop()
{
    // Attempt SCHED_FIFO — fails silently without CAP_SYS_NICE.
    struct sched_param sp{};
    sp.sched_priority = rt_priority_;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);

    static constexpr double kTickDt = 1.0 / 250.0;
    static constexpr double kPosEps = 1e-6;   // m
    static constexpr double kRotEps = 1e-5;   // rad
    static const std::array<double, kNumJoints> kZeros{};

    CmdType prev_cmd_type = CmdType::None;
    std::array<double, kNumJoints> prev_sensor_q{};
    std::array<double, 6> prev_tcp{};
    bool tcp_valid = false;
    bool stream_l_want = false;
    uint64_t stream_j_sends_local = 0;
    uint64_t stream_l_sends_local = 0;

    while (running_.load(std::memory_order_relaxed)) {
        if (!kord_->waitSync(std::chrono::milliseconds(10))) {
            continue;
        }

        // Keep sensor samples fresh even before the first command so Vel-mode
        // entry never anchors vel_q_ref_ on the zero-initialized arrays.
        if (!tcp_valid) {
            rcv_->fetchData();
            prev_sensor_q = rcv_->getJoint(EJV::S_ACTUAL_Q);
            prev_tcp = rcv_->getTCP();
            tcp_valid = true;
            {
                std::lock_guard<std::mutex> lock(state_mtx_);
                current_state_.joint_q  = prev_sensor_q;
                current_state_.tcp_pose = prev_tcp;
                ++current_state_.tick;
                current_state_.alarm_code = rcv_->systemAlarmState();
                current_state_.alarm = (current_state_.alarm_code != 0);
            }
            continue;
        }

        // ── Read pending command ───────────────────────────────────────────────
        CmdType    cmd_type;
        DJCCmd     djc;
        VelCmd     vel;
        StreamJCmd sj;
        bool       sj_dirty;
        StreamLCmd sl;
        bool       sl_dirty;
        MoveJCmd   mj;
        MoveLCmd   ml;
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            cmd_type = active_cmd_type_;
            djc      = pending_djc_;
            vel      = pending_vel_;
            sj       = pending_stream_j_;
            sj_dirty = stream_j_dirty_;
            sl       = pending_stream_l_;
            sl_dirty = stream_l_dirty_;
            mj       = pending_move_j_;
            ml       = pending_move_l_;
        }

        // ── Send command BEFORE fetchData (matches kord API design) ───────────
        switch (cmd_type) {

        case CmdType::Vel: {
            if (prev_cmd_type != CmdType::Vel) {
                vel_q_ref_ = prev_sensor_q;
                ctl_->directJControl(prev_sensor_q, kZeros, kZeros);
            } else {
                for (int i = 0; i < kNumJoints; ++i) {
                    vel_q_ref_[i] += vel.qd[i] * kTickDt;
                    vel_q_ref_[i] = std::clamp(vel_q_ref_[i], vel.q_min[i], vel.q_max[i]);
                }
                ctl_->directJControl(vel_q_ref_, vel.qd, kZeros);
            }
            stream_l_active_ = false;
            stream_l_have_curr_ = false;
            stream_l_want = false;
            break;
        }

        case CmdType::DJC: {
            if (prev_cmd_type != CmdType::DJC) {
                ctl_->directJControl(prev_sensor_q, kZeros, kZeros);
            } else {
                ctl_->directJControl(djc.q, djc.qd, djc.qdd);
            }
            stream_l_active_ = false;
            stream_l_have_curr_ = false;
            stream_l_want = false;
            break;
        }

        case CmdType::StreamJ: {
            if (sj_dirty) {
                if (++stream_j_counter_ >= stream_j_throttle_.load(std::memory_order_relaxed)) {
                    stream_j_counter_ = 0;
                    {
                        std::lock_guard<std::mutex> lock(cmd_mtx_);
                        stream_j_dirty_ = false;
                    }
                    ctl_->moveJ(sj.q,
                                kr2::kord::TT_JS_TARGET_SPEED, sj.tt_val,
                                kr2::kord::BT_TIME, sj.bt_val,
                                kr2::kord::OT_VIAPOINT);
                    ++stream_j_sends_local;
                }
            } else {
                stream_j_counter_ = 0;
            }
            stream_l_active_ = false;
            stream_l_have_curr_ = false;
            stream_l_want = false;
            break;
        }

        case CmdType::StreamL: {
            // Latch a new Python goal; micro-step (or hold) toward it on the
            // throttled send schedule. Stopping between 10 Hz Python updates is
            // what made motion feel jerky — keep streaming until hands-off.
            if (sl_dirty) {
                {
                    std::lock_guard<std::mutex> lock(cmd_mtx_);
                    stream_l_dirty_ = false;
                }
                stream_l_goal_ = sl.tcp;
                stream_l_want = true;
                stream_l_counter_ = 0;
                stream_l_ticks_since_goal_ = 0;
            }
            if (stream_l_want && tcp_valid) {
                // Only re-anchor when we have no commanded pose yet (true start
                // / after idle stop). Mid-stream goal updates continue from
                // stream_l_curr_ so via-points stay continuous.
                if (!stream_l_have_curr_) {
                    stream_l_curr_ = prev_tcp;
                    stream_l_have_curr_ = true;
                }
                stream_l_active_ = true;
                stream_l_want = false;
            }

            if (stream_l_active_) {
                ++stream_l_ticks_since_goal_;
                const int throttle = stream_l_throttle_.load(std::memory_order_relaxed);
                if (++stream_l_counter_ >= throttle) {
                    stream_l_counter_ = 0;
                    // Advance over the full send period so throttle does not
                    // silently slow motion by throttle×.
                    const double send_dt = kTickDt * static_cast<double>(std::max(1, throttle));
                    const bool advancing = advance_pose(
                        stream_l_curr_, stream_l_goal_,
                        sl.max_pos_speed, sl.max_rot_speed, send_dt,
                        kPosEps, kRotEps);
                    ctl_->moveL(stream_l_curr_,
                                sl.tt, sl.tt_val,
                                sl.bt, sl.bt_val,
                                kr2::kord::OT_VIAPOINT);
                    ++stream_l_sends_local;
                    // Hold the landed pose across gaps between Python goals
                    // (~10 Hz). Only stop after a hands-off grace so we do not
                    // flood a static via-point forever (that can drop sessions).
                    static constexpr int kIdleGraceTicks = 50;  // ~200 ms @ 250 Hz
                    if (!advancing && stream_l_ticks_since_goal_ >= kIdleGraceTicks) {
                        stream_l_active_ = false;
                        stream_l_have_curr_ = false;
                    }
                }
            } else {
                stream_l_counter_ = 0;
            }
            break;
        }

        case CmdType::MoveJ:
            ctl_->moveJ(mj.q, mj.tt, mj.tt_val, mj.bt, mj.bt_val, mj.ot);
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                active_cmd_type_ = CmdType::None;
            }
            stream_l_active_ = false;
            stream_l_have_curr_ = false;
            stream_l_want = false;
            break;

        case CmdType::MoveL:
            ctl_->moveL(ml.tcp, ml.tt, ml.tt_val, ml.bt, ml.bt_val, ml.ot);
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                active_cmd_type_ = CmdType::None;
            }
            stream_l_active_ = false;
            stream_l_have_curr_ = false;
            stream_l_want = false;
            break;

        default:
            stream_l_active_ = false;
            break;
        }

        prev_cmd_type = cmd_type;

        // ── Read sensor state AFTER sending command ───────────────────────────
        rcv_->fetchData();
        prev_sensor_q = rcv_->getJoint(EJV::S_ACTUAL_Q);
        prev_tcp = rcv_->getTCP();
        tcp_valid = true;

        uint32_t alarm_code = 0;
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            current_state_.joint_q        = prev_sensor_q;
            current_state_.joint_qd       = rcv_->getJoint(EJV::S_ACTUAL_QD);
            current_state_.joint_tau      = rcv_->getJoint(EJV::S_ACTUAL_TRQ);
            current_state_.tcp_pose       = prev_tcp;
            current_state_.stream_j_sends = stream_j_sends_local;
            current_state_.stream_l_sends = stream_l_sends_local;
            ++current_state_.tick;
            current_state_.alarm_code = rcv_->systemAlarmState();
            current_state_.alarm      = (current_state_.alarm_code != 0);
            alarm_code = current_state_.alarm_code;
        }

        // Alarm: drop the open segment and re-anchor on the next goal from TCP.
        if (alarm_code != 0) {
            stream_l_active_ = false;
            stream_l_have_curr_ = false;
            stream_l_want = false;
        }
    }
}

} // namespace rio
