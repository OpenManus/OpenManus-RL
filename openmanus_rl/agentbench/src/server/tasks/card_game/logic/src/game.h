#pragma once
#include <limits>
#include "player.h"

class Game {
  public:
    enum State {
        READY = 0,
        PICK = 2,
        ASSERT = 3,
        ACTION = 4,
        END = 5
    };
    static constexpr int STATE_LIMIT = 128;
    static std::string state_info(State state);
    int winner = -1;  //  获胜玩家id
    int last_winner = -1;  //  上一轮获胜玩家id
    std::vector<Player> players;
    int state = 1;  //  当前回合(传递给Judger的)
    int last_round_state = 0;
    int gamestate = READY;  //  当前阶段
    int cur_turn = 0;  //  当前操作者
    int first_mover; // 当前轮的先手
    bool over = false;  //  游戏是否结束
    int cnt = 0;
    int imiid[2] = {-1, -1};
    int errorai = 0;
    int score = 0;  //  记录比分
    int rounds = 0;  //  记录战斗轮数
    Game();
    Json::Value to_json() const;
};