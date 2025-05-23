#include "game.h"

//  将 Game::State 转为对应阶段名称
std::string Game::state_info(Game::State state){
    switch(state){
        case READY: return "准备阶段";
        case PICK: return "选择阶段";
        case ASSERT: return "断言阶段";
        case ACTION: return "行动阶段";
        case END: return "游戏结束";
    }
    return "";
}

Game::Game(){
    players.clear();
    players.push_back(Player(0));
    players.push_back(Player(1));
}

/*
    按照成员变量定义顺序转成 JSON
    样例：
        {
            "winner": winner,
            "players": [
                player1,
                player2,
                ...
            ],
            "state": state,
            "gamestate": gamestate,
            "cur_turn": cur_turn,
            "over": over,
            "cnt": cnt,
            "score": score,
            "rounds": rounds
        }
*/
Json::Value Game::to_json() const{
    Json::Value json;
    json["winner"] = winner;
    json["players"].resize(0);
    for(const auto& player : players)
        json["players"].append(player.to_json());
    json["state"] = state;
    json["gamestate"] = gamestate;
    json["cur_turn"] = cur_turn;
    json["over"] = over;
    json["cnt"] = cnt;
    json["score"] = score;
    json["rounds"] = rounds;
    return json;
}