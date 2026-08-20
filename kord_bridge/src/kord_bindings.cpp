#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "kord_bridge.hpp"

namespace nb = nanobind;
using namespace rio;

NB_MODULE(_kord_bridge, m)
{
    nb::enum_<kr2::kord::TrackingType>(m, "TrackingType")
        .value("TT_TIME",            kr2::kord::TT_TIME)
        .value("TT_WS_TARGET_SPEED", kr2::kord::TT_WS_TARGET_SPEED)
        .value("TT_JS_TARGET_SPEED", kr2::kord::TT_JS_TARGET_SPEED);

    nb::enum_<kr2::kord::BlendType>(m, "BlendType")
        .value("BT_TIME",            kr2::kord::BT_TIME)
        .value("BT_WS_ACCELERATION", kr2::kord::BT_WS_ACCELERATION)
        .value("BT_WS_RADIUS",       kr2::kord::BT_WS_RADIUS)
        .value("BT_JS_ACCELERATION", kr2::kord::BT_JS_ACCELERATION);

    nb::class_<KordBridge::State>(m, "State")
        .def_ro("joint_q",        &KordBridge::State::joint_q)
        .def_ro("joint_qd",       &KordBridge::State::joint_qd)
        .def_ro("joint_tau",      &KordBridge::State::joint_tau)
        .def_ro("tcp_pose",       &KordBridge::State::tcp_pose)
        .def_ro("tick",           &KordBridge::State::tick)
        .def_ro("stream_j_sends", &KordBridge::State::stream_j_sends)
        .def_ro("stream_l_sends", &KordBridge::State::stream_l_sends)
        .def_ro("alarm",          &KordBridge::State::alarm)
        .def_ro("alarm_code",     &KordBridge::State::alarm_code);

    nb::class_<KordBridge::DJCCmd>(m, "DJCCmd")
        .def(nb::init<>())
        .def_rw("q",   &KordBridge::DJCCmd::q)
        .def_rw("qd",  &KordBridge::DJCCmd::qd)
        .def_rw("qdd", &KordBridge::DJCCmd::qdd)
        .def_rw("tau", &KordBridge::DJCCmd::tau);

    nb::class_<KordBridge::VelCmd>(m, "VelCmd")
        .def(nb::init<>())
        .def_rw("qd",    &KordBridge::VelCmd::qd)
        .def_rw("q_min", &KordBridge::VelCmd::q_min)
        .def_rw("q_max", &KordBridge::VelCmd::q_max);

    nb::class_<KordBridge::StreamJCmd>(m, "StreamJCmd")
        .def(nb::init<>())
        .def_rw("q",      &KordBridge::StreamJCmd::q)
        .def_rw("tt_val", &KordBridge::StreamJCmd::tt_val)
        .def_rw("bt_val", &KordBridge::StreamJCmd::bt_val);

    nb::class_<KordBridge::StreamLCmd>(m, "StreamLCmd")
        .def(nb::init<>())
        .def_rw("tcp",    &KordBridge::StreamLCmd::tcp)
        .def_rw("tt",     &KordBridge::StreamLCmd::tt)
        .def_rw("tt_val", &KordBridge::StreamLCmd::tt_val)
        .def_rw("bt",     &KordBridge::StreamLCmd::bt)
        .def_rw("bt_val", &KordBridge::StreamLCmd::bt_val);

    nb::class_<KordBridge::MoveJCmd>(m, "MoveJCmd")
        .def(nb::init<>())
        .def_rw("q",      &KordBridge::MoveJCmd::q)
        .def_rw("tt_val", &KordBridge::MoveJCmd::tt_val)
        .def_rw("bt_val", &KordBridge::MoveJCmd::bt_val);

    nb::class_<KordBridge::MoveLCmd>(m, "MoveLCmd")
        .def(nb::init<>())
        .def_rw("tcp",    &KordBridge::MoveLCmd::tcp)
        .def_rw("tt_val", &KordBridge::MoveLCmd::tt_val)
        .def_rw("bt_val", &KordBridge::MoveLCmd::bt_val);

    nb::class_<KordBridge>(m, "KordBridge")
        .def(nb::init<const std::string&, unsigned int, unsigned int, int>(),
             nb::arg("ip"),
             nb::arg("port")        = 7582u,
             nb::arg("session_id")  = 1u,
             nb::arg("rt_priority") = 80)
        .def("connect",         &KordBridge::connect)
        .def("start",           &KordBridge::start)
        .def("stop",            &KordBridge::stop)
        .def("is_running",      &KordBridge::is_running)
        .def("get_state",       &KordBridge::get_state)
        .def("set_djc_command",      &KordBridge::set_djc_command)
        .def("set_vel_command",      &KordBridge::set_vel_command)
        .def("set_stream_j_command", &KordBridge::set_stream_j_command)
        .def("set_stream_j_throttle", &KordBridge::set_stream_j_throttle,
             nb::arg("n"),
             "Send moveJ every n waitSync ticks. n=1 → full sync rate, n=4 → 31 Hz at 125 Hz sync.")
        .def("set_stream_l_command", &KordBridge::set_stream_l_command)
        .def("set_stream_l_throttle", &KordBridge::set_stream_l_throttle,
             nb::arg("n"),
             "Send moveL every n waitSync ticks. n=1 → full sync rate.")
        .def("queue_move_j",         &KordBridge::queue_move_j)
        .def("queue_move_l",         &KordBridge::queue_move_l);
}
