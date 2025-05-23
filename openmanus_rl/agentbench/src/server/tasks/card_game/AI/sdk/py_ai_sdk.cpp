// 对应的 C++ AI SDK 版本：b055c46a5ccfebbda597238ad74c407dfb69bf56

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ai_client.hpp"
#include "py_json_cast.hpp"

namespace py = pybind11;
using namespace py::literals;

class PyAIClient : public AIClient {
   public:
    using AIClient::AIClient;

    virtual std::vector<int> Pick(Game game) override {
        PYBIND11_OVERRIDE(std::vector<int>, AIClient, Pick, game);
    }
    virtual std::pair<int, int> Assert(Game game) override {
        using Pair = std::pair<int, int>;
        PYBIND11_OVERRIDE(Pair, AIClient, Assert, game);
    }
    virtual Action Act(Game game) override {
        PYBIND11_OVERRIDE(Action, AIClient, Act, game);
    }
};  // class PyAIClient

PYBIND11_MODULE(ai_client, m) {
    m.doc() = "AquaWar AI SDK for Python";

    py::enum_<fish_type>(m, "fish_type")
        .value("spray", fish_type::spray, "射水鱼")
        .value("flame", fish_type::flame, "喷火鱼")
        .value("eel", fish_type::eel, "电鳗")
        .value("sunfish", fish_type::sunfish, "翻车鱼")
        .value("barracuda", fish_type::barracuda, "梭子鱼")
        .value("mobula", fish_type::mobula, "蝠鲼")
        .value("turtle", fish_type::turtle, "海龟")
        .value("octopus", fish_type::octopus, "章鱼")
        .value("whiteshark", fish_type::whiteshark, "大白鲨")
        .value("hammerhead", fish_type::hammerhead, "锤头鲨")
        .value("clownfish", fish_type::clownfish, "小丑鱼")
        .value("imitator", fish_type::imitator, "拟态鱼")
        .export_values();
    // enum fish_type

    py::enum_<active_skill>(m, "active_skill")
        .value("range_attack", active_skill::range_attack,
               "射水鱼(spray)+电鳗(eel)")
        .value("friend_attack", active_skill::friend_attack,
               "喷火鱼(flame)+翻车鱼(sunfish)")
        .value("critical_attack", active_skill::critical_attack,
               "梭子鱼(barracuda)")
        .value("reduce_injury", active_skill::reduce_injury,
               "蝠鲼(mobula)+章鱼(octupus)")
        .value("limit_critical", active_skill::limit_critical, "海龟(turtle)")
        .value("weak_critical", active_skill::weak_critical,
               "大白鲨(whiteshark)+锤头鲨(hammerhead)")
        .value("injury_tansfer", active_skill::injury_tansfer,
               "小丑鱼(clownfish)")
        .export_values();
    // enum active_skill

    py::enum_<passive_skill>(m, "passive_skill")
        .value("injury_back", passive_skill::injury_back,
               "射水鱼(spray)+喷火鱼(flame)")
        .value("friend_injury_transfer", passive_skill::friend_injury_transfer,
               "电鳗(eel)+翻车鱼(sunfish)")
        .value("avoid_injury", passive_skill::avoid_injury,
               "梭子鱼(barracuda)+蝠鲼(mobula)")
        .value("shield", passive_skill::shield, "海龟(turtle)")
        .value("limit_heal", passive_skill::limit_heal,
               "章鱼(octopus)+大白鲨(whiteshark)")
        .value("dead_boom", passive_skill::dead_boom, "锤头鲨(hammerhead)")
        .value("injury_transfer", passive_skill::injury_transfer,
               "小丑鱼(clownfish)")
        .export_values();
    // enum passive_skill

    py::class_<Fish>(m, "Fish")
        .def_readwrite("id", &Fish::id)
        .def_readwrite("type", &Fish::type)
        .def_readwrite("hp", &Fish::hp)
        .def_readwrite("atk", &Fish::atk)
        .def_readwrite("skill_used", &Fish::skill_used)
        .def_readwrite("active", &Fish::active)
        .def_readwrite("passive", &Fish::passive)
        .def(py::init())
        .def(py::init<int, int, int>(), "id"_a, "hp"_a, "atk"_a)
        .def("__copy__", [](const Fish& fish) -> Fish { return fish; })
        .def(
            "__deepcopy__",
            [](const Fish& fish, py::dict) -> Fish { return fish; }, "memo"_a);
    // class Fish

    py::enum_<skill_type>(m, "skill_type")
        .value("aoe", skill_type::aoe)
        .value("infight", skill_type::infight)
        .value("crit", skill_type::crit)
        .value("subtle", skill_type::subtle)
        .value("normalattack", skill_type::normalattack)
        .export_values();
    // enum skill_type

    py::enum_<passive_type>(m, "passive_type")
        .value("counter", passive_type::counter)
        .value("deflect", passive_type::deflect)
        .value("reduce", passive_type::reduce)
        .value("heal", passive_type::heal)
        .value("explode", passive_type::explode)
        .export_values();
    // enum passive_type

    m.attr("max_length") = 24;

    py::class_<ActionInfo>(m, "ActionInfo")
        .def_readwrite("is_skill", &ActionInfo::is_skill, "是否是技能")
        .def_readwrite("action_fish", &ActionInfo::action_fish, "行动的鱼")
        .def_readwrite("num_friend_targets", &ActionInfo::num_friend_targets,
                       "友方目标个数")
        .def_readwrite("num_enemy_targets", &ActionInfo::num_enemy_targets,
                       "敌方目标个数")
        .def_readwrite("friend_targets", &ActionInfo::friend_targets,
                       "友方目标")
        .def_readwrite("enemy_targets", &ActionInfo::enemy_targets, "敌方目标")
        .def_readwrite("friend_excepted_injury",
                       &ActionInfo::friend_excepted_injury, "友方预期伤害")
        .def_readwrite("enemy_excepted_injury",
                       &ActionInfo::enemy_excepted_injury, "敌方预期伤害")
        .def_readwrite("friend_expected_injury",
                       &ActionInfo::friend_expected_injury, "友方预期伤害")
        .def_readwrite("enemy_expected_injury",
                       &ActionInfo::enemy_expected_injury, "敌方预期伤害")
        .def_readwrite("num_friend_injury", &ActionInfo::num_friend_injury,
                       "友方受伤鱼的数量")
        .def_readwrite("num_enemy_injury", &ActionInfo::num_enemy_injury,
                       "敌方受伤鱼的数量")
        .def_readwrite("friend_injury_id", &ActionInfo::friend_injury_id,
                       "友方受伤鱼的位置编号")
        .def_readwrite("enemy_injury_id", &ActionInfo::enemy_injury_id,
                       "敌方受伤鱼的位置编号")
        .def_readwrite("friend_injury", &ActionInfo::friend_injury,
                       "友方实际伤害")
        .def_readwrite("enemy_injury", &ActionInfo::enemy_injury,
                       "敌方实际伤害")
        .def_readwrite("friend_injury_timestamp",
                       &ActionInfo::friend_injury_timestamp, "友方受伤时间戳")
        .def_readwrite("enemy_injury_timestamp",
                       &ActionInfo::enemy_injury_timestamp, "敌方受伤时间戳")
        .def_readwrite("friend_injury_traceable",
                       &ActionInfo::friend_injury_traceable,
                       "友方受伤是否可追踪")
        .def_readwrite("enemy_injury_traceable",
                       &ActionInfo::enemy_injury_traceable,
                       "敌方受伤是否可追踪")
        .def_readwrite("type", &ActionInfo::type)
        .def_readwrite("num_friend_passives", &ActionInfo::num_friend_passives,
                       "友方被动个数")
        .def_readwrite("num_enemy_passives", &ActionInfo::num_enemy_passives,
                       "敌方被动个数")
        .def_readwrite("friend_passives_id", &ActionInfo::friend_passives_id,
                       "友方触发被动的鱼的id")
        .def_readwrite("enemy_passives_id", &ActionInfo::enemy_passives_id,
                       "敌方触发被动的鱼的id")
        .def_readwrite("friend_types", &ActionInfo::friend_types,
                       "友方触发被动类型")
        .def_readwrite("enemy_types", &ActionInfo::enemy_types,
                       "敌方触发被动类型")
        .def_readwrite("friend_passive_value",
                       &ActionInfo::friend_passive_value, "友方减伤比/回血量")
        .def_readwrite("enemy_passive_value", &ActionInfo::enemy_passive_value,
                       "敌方减伤比/回血量")
        .def_readwrite("friend_passives_timestamp",
                       &ActionInfo::friend_passives_timestamp,
                       "友方触发被动时间戳")
        .def_readwrite("enemy_passives_timestamp",
                       &ActionInfo::enemy_passives_timestamp,
                       "敌方触发被动时间戳")
        .def(py::init())
        .def("clear_skill", &ActionInfo::clear_skill)
        .def("clear_hit", &ActionInfo::clear_hit)
        .def("clear_passive", &ActionInfo::clear_passive)
        .def("clear", &ActionInfo::clear)
        .def("is_empty", &ActionInfo::is_empty);
    // class ActionInfo

    py::class_<AssertInfo>(m, "AssertInfo")
        .def_readwrite("assertPos", &AssertInfo::assertPos)
        .def_readwrite("assertContent", &AssertInfo::assertContent)
        .def_readwrite("assertResult", &AssertInfo::assertResult)
        .def(py::init())
        .def("clear", &AssertInfo::clear)
        .def("is_empty", &AssertInfo::is_empty);
    // class AssertInfo

    py::class_<Game>(m, "Game", "游戏局面")
        .def_readonly("my_fish", &Game::my_fish)
        .def_readonly("enemy_fish", &Game::enemy_fish)
        .def_readonly("avatar_id", &Game::avatar_id)
        .def_readonly("first_mover", &Game::first_mover)
        .def_readonly("enemy_action", &Game::enemy_action)
        .def_readonly("my_action", &Game::my_action)
        .def_readonly("enemy_assert", &Game::enemy_assert)
        .def_readonly("my_assert", &Game::my_assert)
        .def_readonly("round", &Game::round)
        .def_readonly("round1_win", &Game::round1_win)
        .def_readonly("round2_win", &Game::round2_win)
        .def_readonly("round3_win", &Game::round3_win)
        .def_readonly("last_round_finish_reason",
                      &Game::last_round_finish_reason)
        .def_readonly("state_limit_exceed", &Game::state_limit_exceed)
        .def(py::init<Fish, Fish, Fish, Fish, Fish, Fish, Fish, Fish, int, bool,
                      int, ActionInfo, ActionInfo, AssertInfo, AssertInfo, bool,
                      bool, bool, int, bool>(),
             "friend1"_a, "friend2"_a, "friend3"_a, "friend4"_a, "enemy1"_a,
             "enemy2"_a, "enemy3"_a, "enemy4"_a, "avatar_id"_a, "first_mover"_a,
             "round"_a, "enemy_action"_a, "my_action"_a, "enemy_assert"_a,
             "my_assert"_a, "round1_win"_a, "round2_win"_a, "round3_win"_a,
             "last_round_finish_reason"_a, "state_limit_exceed"_a);
    // class Game

    py::enum_<err_type>(m, "err_type")
        .value("err_target", err_type::err_target)
        .value("err_target_out_of_range", err_type::err_target_out_of_range)
        .value("err_action_type", err_type::err_action_type)
        .value("err_action_fish", err_type::err_action_fish)
        .value("err_action_not_choose", err_type::err_action_not_choose)
        .value("err_fish_not_choose", err_type::err_fish_not_choose)
        .value("success", err_type::success)
        .export_values();
    // enum err_type

    py::class_<Action>(m, "Action")
        .def(py::init<Game>(), "game"_a)
        .def("set_action_type", &Action::set_action_type, "action_type"_a)
        .def("set_action_fish", &Action::set_action_fish, "fish_id"_a)
        .def("set_enemy_target", &Action::set_enemy_target, "enemy_id"_a)
        .def("set_friend_target", &Action::set_friend_target, "friend_id"_a)
        .def("get_action_type", &Action::get_action_type)
        .def("get_action_fish", &Action::get_action_fish)
        .def("get_enemy_target", &Action::get_enemy_target)
        .def("get_friend_target", &Action::get_friend_target);
    // class Action

    py::class_<AIClient, PyAIClient>(m, "AIClient")
        .def(py::init())
        .def("random", &AIClient::random, "l"_a, "r"_a)
        .def("debug_msg", &AIClient::debug_msg, "str"_a)
        .def("clear_msg", &AIClient::clear_msg)
        .def("sendLen", &AIClient::sendLen, "s"_a,
             "发送消息前需要先发送长度\n"
             "调用sendLen函数")
        .def("sendrecv_msg", &AIClient::sendrecv_msg, "s"_a, "发送消息")
        .def("listen", &AIClient::listen, "接收消息")
        .def("get_my_remain_fishes", &AIClient::get_my_remain_fishes,
             "获得自己剩余的鱼的编号")
        .def("get_enemy_id", &AIClient::get_enemy_id, "pos"_a,
             "获得敌方一条鱼的id")
        .def("get_enemy_hp", &AIClient::get_enemy_hp, "pos"_a,
             "获得敌方一条鱼的hp")
        // .def("get_enemy_atk", &AIClient::get_enemy_atk, "pos"_a,
        //      "获得敌方一条鱼的atk")
        .def("get_my_id", &AIClient::get_my_id, "pos"_a)
        .def("get_my_hp", &AIClient::get_my_hp, "pos"_a, "获得我方一条鱼的hp")
        .def("get_my_atk", &AIClient::get_my_atk, "pos"_a,
             "获得我方一条鱼的atk")
        .def("get_lowest_health_enemy", &AIClient::get_lowest_health_enemy)
        .def("get_lowest_health_enemies", &AIClient::get_lowest_health_enemies)
        .def("get_enemy_living_fishes", &AIClient::get_enemy_living_fishes,
             "获得敌方剩余存活的鱼的编号")
        .def("get_my_living_fishes", &AIClient::get_my_living_fishes,
             "获得自己剩余存活的鱼的编号")
        .def("get_my_living_ally", &AIClient::get_my_living_ally, "pos"_a,
             "获得自己一条鱼的其余存活队友")
        .def("get_avatar_id", &AIClient::get_avatar_id,
             "获得己方拟态鱼本轮模仿的鱼的编号")
        .def("get_first_mover", &AIClient::get_first_mover,
             "获取己方这一轮是否是先手")
        .def("get_skill_used", &AIClient::get_skill_used, "pos"_a,
             "获得自己一条鱼的主动技能使用次数")
        .def("auto_valid_action", &AIClient::auto_valid_action, "pos"_a,
             "action"_a,
             "自动为一个位置的鱼选取合法行动目标，优先高概率使用主动技能，若不"
             "可能合法则使用普通攻击")
        .def("Pick", &AIClient::Pick, "game"_a,
             "用户编写AI的函数\n用于完成选鱼上场的操作")
        .def("Assert", &AIClient::Assert, "game"_a,
             "用户编写AI的函数\n用于断言操作\n第一个数字返回-1表示不使用断言")
        .def("Act", &AIClient::Act, "game"_a,
             "用户编写AI的函数\n用于行动操作，需要返回一个int类型的二维向量，该"
             "向量必须有四行\n"
             "第一个向量的第一个数字表示行动类型\n第二个向量的第一个数字表示友"
             "方鱼的位置\n"
             "若type == 0：\n"
             "    第三个向量的第一个数字代表需要攻击的敌方鱼位置\n"
             "若type == 1：\n"
             "    第三个向量代表友方鱼目标\n"
             "    第四个向量代表敌方鱼作用目标\n"
             "返回值是行动类型与选取的友方鱼类\n"
             "\n"
             "示例：\n"
             "    action = [[0 for _ in range(4)] for _ in range(4)]\n"
             "然后只需以下标修改各向量内容即可")
        .def("parseGameInfo", &AIClient::parseGameInfo, "gameInfo"_a)
        .def("parseEnemyActionInfo", &AIClient::parseEnemyActionInfo,
             "actionInfo"_a)
        .def("parseMyActionInfo", &AIClient::parseMyActionInfo, "actionInfo"_a)
        .def("parseEnemyAssert", &AIClient::parseEnemyAssert, "assertInfo"_a)
        .def("parseMyAssert", &AIClient::parseMyAssert, "assertInfo"_a)
        .def("updateGame", &AIClient::updateGame, "更新函数")
        .def("Action_Pick", &AIClient::Action_Pick)
        .def("Action_Assert", &AIClient::Action_Assert)
        .def("Action_Action", &AIClient::Action_Action)
        .def("Action_Finish", &AIClient::Action_Finish)
        .def("run", &AIClient::run, "回合制循环");
    // class AIClient
}  // py_module ai_client
