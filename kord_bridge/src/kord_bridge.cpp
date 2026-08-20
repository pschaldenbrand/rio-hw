#include "kord_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <pthread.h>

using namespace kr2::kord;
using EJV = ReceiverInterface::EJointValue;

namespace rio {

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

void KordBridge::rt_loop()
{
    // Attempt SCHED_FIFO — fails silently without CAP_SYS_NICE.
    struct sched_param sp{};
    sp.sched_priority = rt_priority_;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);

    static constexpr double kTickDt = 1.0 / 250.0;
    static const std::array<double, kNumJoints> kZeros{};

    CmdType prev_cmd_type = CmdType::None;
    std::array<double, kNumJoints> prev_sensor_q{};
    uint64_t stream_j_sends_local = 0;  // RT-thread-local; copied into state under lock each tick
    uint64_t stream_l_sends_local = 0;

    while (running_.load(std::memory_order_relaxed)) {
        if (!kord_->waitSync(std::chrono::milliseconds(10))) {
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
        // waitSync signals that the robot is ready for the next command.
        // Sending directJControl immediately keeps us within the robot's
        // command window. Calling fetchData first delays the command by one
        // processing pass and causes torque deviation faults.
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
            break;
        }

        case CmdType::DJC: {
            if (prev_cmd_type != CmdType::DJC) {
                ctl_->directJControl(prev_sensor_q, kZeros, kZeros);
            } else {
                ctl_->directJControl(djc.q, djc.qd, djc.qdd);
            }
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
            break;
        }

        case CmdType::StreamL: {
            // Send only when Python publishes a new TCP target. Continuous
            // resend at high rate has dropped KORD sessions on this robot.
            if (sl_dirty) {
                if (++stream_l_counter_ >= stream_l_throttle_.load(std::memory_order_relaxed)) {
                    stream_l_counter_ = 0;
                    {
                        std::lock_guard<std::mutex> lock(cmd_mtx_);
                        stream_l_dirty_ = false;
                    }
                    ctl_->moveL(sl.tcp,
                                sl.tt, sl.tt_val,
                                sl.bt, sl.bt_val,
                                kr2::kord::OT_VIAPOINT);
                    ++stream_l_sends_local;
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
            break;

        case CmdType::MoveL:
            ctl_->moveL(ml.tcp, ml.tt, ml.tt_val, ml.bt, ml.bt_val, ml.ot);
            {
                std::lock_guard<std::mutex> lock(cmd_mtx_);
                active_cmd_type_ = CmdType::None;
            }
            break;

        default:
            break;
        }

        prev_cmd_type = cmd_type;

        // ── Read sensor state AFTER sending command ───────────────────────────
        rcv_->fetchData();
        prev_sensor_q = rcv_->getJoint(EJV::S_ACTUAL_Q);

        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            current_state_.joint_q        = prev_sensor_q;
            current_state_.joint_qd       = rcv_->getJoint(EJV::S_ACTUAL_QD);
            current_state_.joint_tau      = rcv_->getJoint(EJV::S_ACTUAL_TRQ);
            current_state_.tcp_pose       = rcv_->getTCP();
            current_state_.stream_j_sends = stream_j_sends_local;
            current_state_.stream_l_sends = stream_l_sends_local;
            ++current_state_.tick;
            current_state_.alarm_code = rcv_->systemAlarmState();
            current_state_.alarm      = (current_state_.alarm_code != 0);
        }
    }
}

} // namespace rio
