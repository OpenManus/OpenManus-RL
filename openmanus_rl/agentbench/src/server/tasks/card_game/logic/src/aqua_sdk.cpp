#include <fstream>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <ctime>
#include <random>
#include <optional>
#include <future>
#include "jsoncpp/json/json.h"
#include "game.h"
#include "fish.h"

static std::string convert_to_byte(std::string msg){
    int len = msg.length();
    char lenb[4];
    for (int i = 0; i < 4; i ++)
        lenb[3 - i] = (unsigned char) (len >> (8 * i));
    std::string res(lenb);
    res += msg;
    return res;
}

enum ErrorType{                //错误类型
    judger_error,
    parse_failure,
    player_re,
    player_tle,
    type_error,
    key_missing,
    value_error,    
    range_error,
    choice_repeat,

    repeat_finish,
    pick_number_error,
    pick_dead_fish,

    assert_dead_fish,
    assert_exposed_fish,

    action_with_dead_fish,
    action_rules_error
};

class AquaWarSDK {
  private:
    enum class RoundEndFlag {NormalAssert, NormalAction, SLEAssert, SLEAction}; // 每轮结束的原因(区分断言阶段结束和行动阶段结束)

    Json::Reader reader;
    Json::Value root;  // xwb: 目前没使用过
    Json::Value state; // szh:每一回合的状态以及动态操作
    Json::StreamWriterBuilder w_builder;
    std::unique_ptr<Json::StreamWriter> writer;
    Json::Value replay;  // 记录回放文件
    std::ofstream fout;  //  replay
    FILE* f;  // 运行时输出信息对应文件
    Game game;
    std::ostringstream os;
    std::string Msg;
    std::ofstream info;
    std::ofstream debugger;
    bool acted; //玩家是否已经行动过
    std::optional<Json::Value> action_info;   //玩家的行动信息(以及是否合法)
    Json::Value last_assert;                  //上一次的断言信息
    std::vector<Json::Value> last_action;     //上一次双方的行动信息

  public:
    AquaWarSDK() {
        game = Game();
        // f = fopen("data.out", "w+");
        writer = std::unique_ptr<Json::StreamWriter>
                            (w_builder.newStreamWriter());
        //info.open("info.txt", std::ios::out);                    
        //debugger.open("debug_info.txt", std::ios::out);
        //debugger << "======= Debug Info =======" << std::endl;
        //debugger.close();
        last_action.resize(2);
    }

    std::string remove_enter(std::string origin)
    {
        std::string msg = origin;
        msg.erase(remove(msg.begin(), msg.end(), '\n'), msg.end());
	    msg.erase(remove(msg.begin(), msg.end(), '\r'), msg.end());
	    msg.erase(remove(msg.begin(), msg.end(), '\t'), msg.end());
	    return msg;
    }

    void debug_msg(std::string str){
        return;
        debugger.open("debug_info.txt", std::ios::out | std::ios::app);
        debugger << str << std::endl;
        debugger.close();
    }

    void info_msg(std::string str){
        return;
        info << str;
    }

    void data_msg(std::string str){
        return;
        fprintf(f, str.c_str());
    }

    /*
        sendMsg内发送消息前需要先发送长度
        调用该函数发送
    */
    void sendLen(size_t s){
        int len = s;
        unsigned char lenb[4];
        lenb[0] = (unsigned char)(len);
        lenb[1] = (unsigned char)(len >> 8);
        lenb[2] = (unsigned char)(len >> 16);
        lenb[3] = (unsigned char)(len >> 24);
        for (int i = 0; i < 4; i++)
            printf("%c", lenb[3 - i]);  // 发给Judger
    }

    bool checkLen(size_t len) {
        for (int i = 0; i < 4; i++) {
            unsigned char x = (len >> (i * 8));
            if (x == '\n' || x == '\r') return false;
        }
        return true;
    }

    void cleanse(std::string &msg) {
        int targetLen = msg.length();
        while (!checkLen(targetLen)) ++targetLen;
        int diff = targetLen - msg.length();
        if (diff) msg.append(std::string(diff, ' '));
    }

    /*
        向Judger发送消息
        需要在前面加上数据长度作为数据头
    */
    void sendMsg(std::string msg)
    {
        //  在包前加入数据头
        std::string s = remove_enter(msg);
        cleanse(s);
        debug_msg(std::to_string(s.length()));
        debug_msg(s);
        sendLen(s.length());
        sendLen(-1);
        std::cout << s;
        std::cout.flush();
    }

    //  监听直到收到Judger的消息
    std::string listen(){
        char lenr[4];
        std::string msg;
        while (true){
            scanf("%4c", lenr);
            int len = (unsigned int)((((unsigned int)lenr[3]) & 255) |
                                    ((((unsigned int)lenr[2]) & 255) << 8) |
                                    ((((unsigned int)lenr[1]) & 255) << 16) |
                                    ((((unsigned int)lenr[0]) & 255) << 24));
            for (int i = 0; i < len; i ++)
                msg += getchar();
            break;
        }
        return msg;
    }

    /*
        交互开始时接收judger发来的初始信息
        通信格式: json
        通信内容: 启动的AI状态
                 0:启动失败 1:启动成功 2:播放器
                 AI的数量
                 replay的路径
        通信样例:
                {
                    "player_list": [1, 2]
                    "player_num": 2
                    "replay": "usr/bin/replay.json"
                }
    */
    void receive_player_list(){
        std::string msg = listen();
        Json::Value root;
        if (!reader.parse(msg, root)) {
            //fprintf(f, "fail to parse root\n");

            data_msg("fail to parse root\n");
            fclose(f);
            exit(0);
        }
        if (root["player_num"].asInt() != 2){
            /*fprintf(f, "The game should only have two players\n");
            fprintf(f, "found %d.", root["player_num"].asInt());*/

            data_msg("The game should only have two players\n");
            data_msg("found " + root["player_num"].asString() + ".");
            fclose(f);
            exit(0);
        }
        auto& ls = root["player_list"];
        fout.open(root["replay"].asString());
        this->game = Game();
        int n = ls.size();
        for (int i = 0; i < n; i++){
            if (ls[i] == 2)  // 如果是播放器
                game.players[i].type = 2;
        }
    }

    /*
        接收到Judger的初始化信息后，向Judger发送初始化信息
        通信格式: json
        通信内容: 回合时间限制
                 最大消息长度
        通信样例:
                {
                    "state": 0
                    "time": 3,
                    "length": 1024
                },
    */
    void write_limit(int time_limit){
        Json::Value root;
        root["state"] = 0;
        root["time"] = time_limit;
        root["length"] = 1024;
        writer->write(root, &os);
        Msg = os.str();
        os.str("");  // 清空串流
        sendMsg(Msg);
        Msg = "";
    }

    //  游戏开始，先进行初始化交互
    void start(){
        receive_player_list();
    }

    /*
        发送消息
        通信格式: json
        通信内容: 回合数
                 需要发送消息的选手id列表
                 按照上面的列表依次需要发送的数据包
                 本回合需要行动的选手id列表(judger只会转发其的消息)
        通信样例:
                {
                    "state": 233
                    "player": [0, 1]  // 0 to n-1
                    "content":
                    "
                        {   // 传给玩家1的
                            "Action": "Null",
                            "State": "Assert",
                            "fishs": [
                                        [
                                            {
                                                "hp": 100,
                                                "atk": 50,
                                                "def": 20,
                                                "type": 0,
                                                "id": 0,
                                            },
                                            ...
                                        ],
                                        [...],
                                     ],
                            ...
                        }
                    ",
                    "
                        {   // 传给玩家2的
                            "Action": "Assert",
                            "State": "Assert",
                            "fishs": ...
                            ...
                        }

                    ",
                    "listen": 1
                }

    */
    void action_info_modifier(Json::Value& action_info,int curturn){
        if (action_info.size() == 0) return;
        debug_msg("modifier start");
        // debug_msg(std::to_string(action_info.size()));
        if (action_info.isMember("skill")){
            debug_msg("skill modify start");
            for (int i=0;i<(int)action_info["skill"]["targets"].size();i++){
                int player = stoi(action_info["skill"]["targets"][i]["player"].asString());
                action_info["skill"]["targets"][i]["isEnemy"] = (player == curturn ? false : true);
            }
            debug_msg("skill modify end");
        }
        if (action_info.isMember("hit")){
            debug_msg("hit modify start");
            for (int i=0;i<(int)action_info["hit"].size();i++){
                int player = stoi(action_info["hit"][i]["player"].asString());
                action_info["hit"][i]["isEnemy"] = (player == curturn ? false : true);
            }
            debug_msg("hit modify end");
        }
        if (action_info.isMember("passive")){
            debug_msg("passive modify start");
            for (int i = 0; i < (int)action_info["passive"].size(); i++)
            {
                int player = stoi(action_info["passive"][i]["player"].asString());
                action_info["passive"][i]["isEnemy"] = (player == curturn ? false : true);
            }
            debug_msg("passive modify end");
        }
    }

    void write_json(){
        debug_msg("write start");
        debug_msg("now state: " + std::to_string(game.gamestate));
        // xwb(2021.1.21): 仅用于确定先后手
        if (game.gamestate == Game::READY) return;
        { // xwb(2021.1.21): PICK 改为两回合
            if (game.players[game.cur_turn].type == 2)
                write_limit(300);
            else
                write_limit(300);
        }
        Json::Value root;
        root["state"] = game.state;
        if (game.gamestate == Game::PICK){
            int j = game.cur_turn; { // xwb(2021.1.21): 曾经是 for
                root["listen"].append(j);
                root["player"].append(j);
                Json::Value player;
                player["Action"] = "Pick";
                game.players[j].fight_fish.clear();  //  清空上场战斗的鱼，准备新一场战斗
                auto fishs = game.players[j].get_fishs();
                for(auto fish : fishs)
                    player["RemainFishs"].append(fish->id);
                player["FirstMover"] = (game.first_mover == j ? 1 : 0);
                root["content"].append(remove_enter(player.toStyledString()));
            }
        }
        else if (game.gamestate == Game::ASSERT){
            debug_msg("write assert start");
            root["listen"].append(game.cur_turn);
            root["player"].append(game.cur_turn);
            Json::Value player;
            player["Action"] = "Assert";
            player["EnemyAssert"] = last_assert;
            if (last_action.size() == 2){
                debug_msg("write assert enemy assert info start");
                for (int j=0;j<2;j++){
                    action_info_modifier(last_action[j], game.cur_turn);
                }
                debug_msg("write assert enemy assert info end");
                player["EnemyAction"] = last_action[game.cur_turn ^ 1];
                player["MyAction"] = last_action[game.cur_turn];
                debug_msg("write assert action info end");
            }
            Json::Value game_info;
            auto enemy_fishs = game.players[game.cur_turn ^ 1].fight_fish.get_fishs();
            for (auto fish : enemy_fishs)
            {
                if (fish->is_expose)
                    game_info["EnemyFish"].append(fish->id);
                else
                    game_info["EnemyFish"].append(-1);
                if (fish->state == Fish::DEAD)
                    game_info["EnemyHP"].append(0);
                else
                    game_info["EnemyHP"].append(fish->hp);
                /*if (fish->is_expose)
                    game_info["EnemyATK"].append(fish->atk);
                else
                    game_info["EnemyATK"].append(-1);*/
                //game_info["EnemyATK"].append(fish->get_atk());
            }
            auto my_fishs = game.players[game.cur_turn].fight_fish.get_fishs();
            for (auto fish : my_fishs)
            {
                if (fish->is_expose)
                    game_info["MyFish"].append(-fish->id);
                else
                    game_info["MyFish"].append(fish->id);
                if (fish->state == Fish::DEAD)
                    game_info["MyHP"].append(0);
                else
                    game_info["MyHP"].append(fish->hp);
                //game_info["MyATK"].append(fish->atk);
                game_info["MyATK"].append(fish->get_atk());
            }
            player["GameInfo"] = game_info;
            root["content"].append(remove_enter(player.toStyledString()));
            debug_msg("write assert end");
        }
        else if (game.gamestate == Game::ACTION){
            debug_msg("write action start");
            acted = false;
            if (!acted)                   //玩家尚未操作,第一次发送信息
            {
                debug_msg("write action acted");
                root["listen"].append(game.cur_turn);
                root["player"].append(game.cur_turn);
                Json::Value player;
                player["Action"] = "Action";
                player["AssertReply"] = last_assert;
                Json::Value game_info;
                auto enemy_fishs = game.players[game.cur_turn ^ 1].fight_fish.get_fishs();
                for (auto fish : enemy_fishs)
                {
                    if (fish->is_expose)
                        game_info["EnemyFish"].append(fish->id);
                    else
                        game_info["EnemyFish"].append(-1);
                    if (fish->state == Fish::DEAD)
                        game_info["EnemyHP"].append(0);
                    else
                        game_info["EnemyHP"].append(fish->hp);
                    /*if (fish->is_expose)
                        game_info["EnemyATK"].append(fish->atk);
                    else
                        game_info["EnemyATK"].append(-1);*/
                    //game_info["EnemyATK"].append(fish->get_atk());
                }
                auto my_fishs = game.players[game.cur_turn].fight_fish.get_fishs();
                for (auto fish : my_fishs)
                {
                    if (fish->is_expose)
                        game_info["MyFish"].append(-fish->id);
                    else
                        game_info["MyFish"].append(fish->id);
                    if (fish->state == Fish::DEAD)
                        game_info["MyHP"].append(0);
                    else
                        game_info["MyHP"].append(fish->hp);
                    //game_info["MyATK"].append(fish->atk);
                    game_info["MyATK"].append(fish->get_atk());
                }
                player["GameInfo"] = game_info;
                root["content"].append(remove_enter(player.toStyledString()));
            }
            /*else                              //玩家已经操作,第二次发送信息
            {
                // debug_msg("Act!");
                root["listen"].append(game.cur_turn);
                root["player"].append(game.cur_turn);
                Json::Value player;
                player["Action"] = (*acted ? "Success" : "Fail");
                root["content"].append(player.toStyledString());
            }*/
        }
        writer->write(root, &os);
        Msg = os.str();
        os.str("");  // 清空串流
        //fprintf(f, "MSG @ 218: %s\n", Msg.c_str());
        data_msg("MSG @ 218: " + Msg + "\n");
        sendMsg(Msg);
        Msg = "";
    }

    bool pick_msg_checker(Json::Value msg, int player){
        Json::Value error_info;
        error_info["player"] = player;
        if (!msg.isObject()){
            error_info["type"] = ErrorType::type_error;
            state["errors"].append(error_info);
            return false;
        }
        if (!msg.isMember("Action") || !msg.isMember("ChooseFishs")){
            error_info["type"] = ErrorType::key_missing;
            state["errors"].append(error_info);
            return false;
        }
        if (!msg["Action"].isString() || !msg["ChooseFishs"].isArray()){
            error_info["type"] = ErrorType::type_error;
            state["errors"].append(error_info);
            return false;
        }
        if (msg["Action"]!="Pick"){
            error_info["type"] = ErrorType::value_error;
            state["errors"].append(error_info);
            return false;
        }
        if (msg["ChooseFishs"].size()!=4){
            error_info["type"] = ErrorType::pick_number_error;
            state["errors"].append(error_info);
            return false;
        }
        bool flag = true, has_imitation = false;
        std::vector<bool> has_chosen;
        for (int i=0;i<13;i++)
            has_chosen.push_back(false);
        for (int i=0;i<4;i++){
            if (!msg["ChooseFishs"][i].isInt()){
                flag = false;
                error_info["type"] = ErrorType::type_error;
                state["errors"].append(error_info);
                break;
            }
            int id = msg["ChooseFishs"][i].asInt();
            if (id < 1 || id > 12){
                flag = false;
                error_info["type"] = ErrorType::range_error;
                state["errors"].append(error_info);
                break;
            }
            if (has_chosen[id]){
                flag = false;
                error_info["type"] = ErrorType::choice_repeat;
                state["errors"].append(error_info);
                break;
            }
            has_chosen[id] = true;
            if (id == 12)
                has_imitation = true;
        }
        if (!flag)
            return false;
        if (has_imitation){
            if (!msg.isMember("ImitateFish") || !msg["ImitateFish"].isInt()){
                error_info["type"] = (msg.isMember("ImitateFish") ? ErrorType::type_error : ErrorType::key_missing);
                state["errors"].append(error_info);
                return false;
            }
            int imitate = msg["ImitateFish"].asInt();
            if (imitate < 1 || imitate > 11){
                error_info["type"] = ErrorType::range_error;
                state["errors"].append(error_info);
                return false;
            }
        }
        return true;
    }

    bool assert_msg_checker(Json::Value msg, int player){
        Json::Value error_info;
        error_info["player"] = player;
        if (!msg.isObject()){
            error_info["type"] = ErrorType::type_error;
            state["errors"].append(error_info);
            return false;
        }
        if (!msg.isMember("Action")){
            error_info["type"] = ErrorType::key_missing;
            state["errors"].append(error_info);
            return false;
        }
        if (!msg["Action"].isString()){
            error_info["type"] = ErrorType::type_error;
            state["errors"].append(error_info);
            return false;
        }
        if (msg["Action"] != "Null" && msg["Action"] != "Assert"){
            error_info["type"] = ErrorType::value_error;
            state["errors"].append(error_info);
            return false;
        }
        if (msg["Action"] == "Null")
            return true;
        if (!msg.isMember("Pos") || !msg.isMember("ID")){
            error_info["type"] = ErrorType::key_missing;
            state["errors"].append(error_info);
            return false;
        }
        if (!msg["Pos"].isInt() || !msg["ID"].isInt()){
            error_info["type"] = ErrorType::type_error;
            state["errors"].append(error_info);
            return false;
        }
        int pos = msg["Pos"].asInt();
        int id = msg["ID"].asInt();
        if (pos < 0 || pos > 3){
            error_info["type"] = ErrorType::range_error;
            state["errors"].append(error_info);
            return false;
        }
        if (id < 1 || id > 12){
            error_info["type"]= ErrorType::range_error;
            state["errors"].append(error_info);
            return false;
        }
        return true;
    }

    bool action_msg_checker(Json::Value &msg, int player){
        Json::Value error_info;
        error_info["player"] = player;
        if (!msg.isObject()){
            error_info["type"] = ErrorType::type_error;
            state["errors"].append(error_info);
            return false;
        }
        if (!msg.isMember("Type") || !msg["Type"].isInt()){
            error_info["type"] = (msg.isMember("Type") ? ErrorType::type_error : ErrorType::key_missing);
            state["errors"].append(error_info);
            return false;
        }
        if (msg["Type"].asInt() == 0){
            if (!msg.isMember("Action") || !msg.isMember("MyPos") || !msg.isMember("EnemyPos")){
                error_info["type"] = ErrorType::key_missing;
                state["errors"].append(error_info);
                return false;
            }
            if (!msg["Action"].isString() || !msg["Type"].isInt() || !msg["MyPos"].isInt() || !msg["EnemyPos"].isInt()){
                error_info["type"] = ErrorType::type_error;
                state["errors"].append(error_info);
                return false;
            }
            if (msg["Action"]!="Action"){
                error_info["type"] = ErrorType::value_error;
                state["errors"].append(error_info);
                return false;
            }
            int my_pos = msg["MyPos"].asInt();
            int enemy_pos = msg["EnemyPos"].asInt();
            if (my_pos < 0 || my_pos > 3){
                error_info["type"] = ErrorType::range_error;
                state["errors"].append(error_info);
                return false;
            }
            if (enemy_pos < 0 || enemy_pos > 3){
                error_info["type"] = ErrorType::range_error;
                state["errors"].append(error_info);
                return false;
            }
        }
        else{
            if (!msg.isMember("MyList") || msg["MyList"].isNull()) msg["MyList"].resize(0);
            if (!msg.isMember("EnemyList") || msg["EnemyList"].isNull()) msg["EnemyList"].resize(0);
            if (!msg.isMember("Action") || !msg.isMember("MyPos")){
                error_info["type"] = ErrorType::key_missing;
                state["errors"].append(error_info);
                return false;
            }
            if (!msg["Action"].isString() || !msg["MyPos"].isInt()){
                error_info["type"] = ErrorType::type_error;
                state["errors"].append(error_info);
                return false;
            }
            if ((!msg["MyList"].isArray()) || (!msg["EnemyList"].isArray())){
                error_info["type"] = ErrorType::type_error;
                state["errors"].append(error_info);
                return false;
            }
            if (msg["Action"]!="Action"){
                error_info["type"] = ErrorType::value_error;
                state["errors"].append(error_info);
                return false;
            }
            int type = msg["Type"].asInt();
            int my_pos = msg["MyPos"].asInt();
            if (type != 1){
                error_info["type"] = ErrorType::range_error;
                state["errors"].append(error_info);
                return false;
            }
            if (my_pos < 0 || my_pos > 3){
                error_info["type"] = ErrorType::range_error;
                state["errors"].append(error_info);
                return false;
            }
            bool flag = true;
            std::vector<bool> my_chosen;
            std::vector<bool> enemy_chosen;
            for (int i=0;i<4;i++){
                my_chosen.push_back(false);
                enemy_chosen.push_back(false);
            }
            for (int i=0;i<(int)msg["MyList"].size();i++){
                if (!msg["MyList"][i].isInt()){
                    flag = false;
                    error_info["type"] = ErrorType::type_error;
                    state["errors"].append(error_info);
                    break;
                }
                int pos = msg["MyList"][i].asInt();
                if (pos < 0 || pos > 3){
                    flag = false;
                    error_info["type"] = ErrorType::range_error;
                    state["errors"].append(error_info);
                    break;
                }
                if (my_chosen[pos]){
                    flag = false;
                    error_info["type"] = ErrorType::choice_repeat;
                    state["errors"].append(error_info);
                    break;
                }
                my_chosen[pos] = true;
            }
            if (!flag)
                return false;
            for (int i=0;i<(int)msg["EnemyList"].size();i++){
                if (!msg["EnemyList"][i].isInt()){
                    flag = false;
                    error_info["type"] = ErrorType::type_error;
                    state["errors"].append(error_info);
                    break;
                }
                int pos = msg["EnemyList"][i].asInt();
                if (pos < 0 || pos > 3){
                    flag = false;
                    error_info["type"] = ErrorType::range_error;
                    state["errors"].append(error_info);
                    break;
                }
                if (enemy_chosen[pos]){
                    flag = false;
                    error_info["type"] = ErrorType::choice_repeat;
                    state["errors"].append(error_info);
                    break;
                }
                enemy_chosen[pos] = true;
            }
            if (!flag)
                return false;
        }
        return true;
    }
    /*
        接收judger的消息
        因为一个回合可能会有多个选手做出相应,judger每接收一个选手的操作就会把消息转发给逻辑
        逻辑会接收多次消息
        逻辑需要对选手发过来的数据包做格式限制以便于解析,还需处理异常
        注意 : 统一给judger转发字符串类型
        通信格式: json
        通信内容: 回合数
                 选手id
                 数据包
        通信样例:
                {
                    "player": i,
                    "content":
                    "{
                        {
                            "state": 233,
                            "Action": "Assert",
                            "fish_pos": 1,
                            "assert_id": 3,
                        }
                    }"
                }
        异常样例:
                {
                    "player": -1,
                    "content":
                    "{
                        {
                            "player": 4,
                            "error": 0,   // 0 -> collapse ; 1, timeout
                        }
                    }"
    */
   /*
        接收消息后，将玩家的操作(ban,assert,action)记录在state中
   */
    bool read_json(){
        debug_msg("read start");
        game.cnt = 0;
        state["operation"].resize(0);
        bool flag = true;
        int msg_cnt = 0;
        game.errorai = 3;
        int action_err_type = -1;
        static std::vector<int> finish_counter(2);    //统计两方AI发了几次finish
        if(game.gamestate == Game::READY){
            std::fill(finish_counter.begin(), finish_counter.end(),
                      game.last_winner == -1);
            return true;
        }
        while (true){
            debug_msg("read new loop");
            msg_cnt ++;
            if (msg_cnt > 1) break;
            std::string msg = listen();
            /*auto result = std::async([this]() { return this->listen(); });
            auto status = result.wait_for(std::chrono::seconds(game.time_limit()));
            std::string msg = result.get();*/
            Json::Value root;
            if (!reader.parse(msg, root)){     //judger的问题,两边平局处理
                //fprintf(f, "fail to parse root\n");
                data_msg("fail to parse root\n");
                flag = false;
                game.errorai = 3;
                for (int j = 0; j < 2; j ++){
                    Json::Value error_info;
                    error_info["player"] = j;
                    error_info["type"] = ErrorType::judger_error;
                    state["errors"].append(error_info);
                }
                break;
            }
            //fprintf(f, "msg from judger: %s\n", msg.c_str());
            data_msg("msg from judger: " + msg + "\n");

            Json::Value content;
            std::string s_content = root["content"].asString();
            int player_id = stoi(root["player"].asString());

            if (!reader.parse(s_content, content)){  //正常收到玩家消息,但是玩家消息不能解析
                Json::Value error_info;
                error_info["player"] = player_id;
                error_info["type"] = ErrorType::parse_failure;
                state["errors"].append(error_info);
                { // xwb(2021.1.21): 无需特判 PICK
                    flag = false;
                    game.errorai = (1 << player_id);
                    //replay.append(state);
                    break;
                }
            }
            
            if (player_id == -1){         //没有正常收到玩家消息(超时或者意外退出)
                debug_msg("AI error");
                int error = content["error"].asInt();
                int player = content["player"].asInt();
                int contentstate = content["state"].asInt();
                if (error==0){          //AI异常退出
                    Json::Value error_info;
                    error_info["player"] = player;
                    error_info["type"] = ErrorType::player_re;
                    state["errors"].append(error_info);
                    { // xwb(2021.1.21): 无需特判 PICK
                        flag = false;
                        game.errorai = (1 << player);
                        //replay.append(state);
                        break;
                    }
                }
                else{  //AI超时
                    debug_msg("TLE game.state: " + std::to_string(game.state));                   
                    debug_msg("TLE content[player]: " + std::to_string(player));
                    if (contentstate != game.state){
                        msg_cnt--;
                        continue;
                    }
                    { // xwb(2021.1.21): 无需特判 PICK
                        flag = false;
                        game.errorai = (1 << game.cur_turn);
                        Json::Value error_info;
                        error_info["player"] = game.cur_turn;
                        error_info["type"] = ErrorType::player_tle;
                        state["errors"].append(error_info);
                        //replay.append(state);
                        break;
                    }
                }
            }
            /*else 
                game.errorai -= (1 << player_id);*/  //待定:应该不需要?
            info_msg("[ 回合 "+std::to_string(game.state)+" ]：");
            info_msg(Game::state_info(static_cast<Game::State>(game.gamestate)));
            info_msg("\n\t玩家 "+std::to_string(player_id)+" ");
            Json::Value op;
            if (game.gamestate == Game::PICK){
                if (content.isObject() && content.isMember("Action") && content["Action"].isString() && content["Action"].asString() == "Finish"){
                    finish_counter[player_id]++;
                    if (finish_counter[player_id] > 1){
                        Json::Value error_info;
                        error_info["player"] = player_id;
                        error_info["type"] = ErrorType::repeat_finish;
                        state["errors"].append(error_info);
                        if (!flag){
                            game.errorai = 3;
                            break;
                        }
                        else{
                            game.errorai = (1 << player_id);
                            flag = false;
                            replay.append(state);
                            continue;
                        }
                    }
                    --msg_cnt;
                    continue;
                }
                if (!pick_msg_checker(content, player_id)) {
                    if (!flag){
                        game.errorai = 3;
                        break;
                    }
                    else{
                        flag = false;
                        game.errorai = (1 << player_id);
                        //replay.append(state);
                        continue;
                    }
                }
                /*进一步检查玩家有没有挑选之前选过的鱼*/
                bool is_dead;
                for(int i = 0; i < 4; i ++){
                    is_dead = true;
                    int id = stoi(content["ChooseFishs"][i].asString());
                    for (auto fish:game.players[player_id].get_fishs()){
                        if (fish->id == id){
                            is_dead = false;
                            break;
                        }
                    }
                    if (is_dead)
                        break;
                }
                if (is_dead){           //玩家挑选了之前选过的鱼
                    Json::Value error_info;
                    error_info["player"] = player_id;
                    error_info["type"] = ErrorType::pick_dead_fish;
                    state["errors"].append(error_info);
                    if (!flag){
                        game.errorai = 3;
                        break;
                    }
                    else{
                        flag = false;
                        game.errorai = (1 << player_id);
                        //replay.append(state);
                        continue;
                    }
                }

                info_msg("挑选类型编号为：");
                for (int i = 0; i < 4; i ++){
                    int id = stoi(content["ChooseFishs"][i].asString());
                    if (id == 12)
                        game.imiid[player_id] = stoi(content["ImitateFish"].asString());
                    info_msg(std::to_string(id));
                    if (i != 3)
                        info_msg(", ");
                    game.players[player_id].to_war(id);
                }
                info_msg("的鱼出战。\n");
                game.players[player_id].fight_fish.to_fight();  //  改变上场鱼的状态
                for(auto fish :
                    game.players[player_id].fight_fish.get_fishs()) {
                    fish->set_fishset(&(game.players[player_id].fight_fish),
                                      &(game.players[player_id ^ 1].fight_fish));
                    if (fish->id == 12) {
                        auto imifish = dynamic_cast<Imitator*>(fish);
                        imifish->imitate(game.imiid[player_id]);
                    }
                }
                op["Action"] = "Pick";
                op["Fish"] = game.players[player_id].fight_fish.to_json();
                state["operation"].append(op);
                replay.append(state);
                game.cnt += (1 << player_id);
                break;
            }
            else if (game.gamestate == Game::ASSERT){
                debug_msg("read assert start");
                if (!assert_msg_checker(content, player_id)){    //玩家断言信息不合规范
                    flag = false;
                    game.errorai = (1 << player_id);
                    //replay.append(state);    //replay里没有操作内容,只有局面信息
                    break;
                }
                if (content["Action"] == "Null"){
                    info_msg("未进行操作。\n");
                    op["Action"] = "Null";
                    op["ID"] = player_id;
                    state["operation"].append(op);
                    replay.append(state);
                    last_assert["AssertPos"] = Json::nullValue;
                    last_assert["AssertContent"] = Json::nullValue;
                    last_assert["AssertResult"] = Json::nullValue;
                    break;
                }
                if (content["Action"] == "Assert"){
                    auto fishs = game.players[player_id ^ 1].fight_fish.get_fishs();
                    int pos = stoi(content["Pos"].asString());
                    int id = stoi(content["ID"].asString());
                    info_msg("断言对方编号为 "+std::to_string(pos)+" 的鱼类型是 "+std::to_string(id)+"，断言");
                    op["Action"] = "Assert";
                    op["ID"] = player_id;
                    op["Pos"] = pos;
                    op["id"] = id;
                    if (fishs[pos]->state == Fish::DEAD || fishs[pos]->is_expose){  //玩家断言不符合规则
                        Json::Value error_info;
                        error_info["player"] = player_id;
                        error_info["type"] = (fishs[pos]->state == Fish::DEAD ? ErrorType::assert_dead_fish : ErrorType::assert_exposed_fish);
                        state["errors"].append(error_info);
                        game.errorai = (1 << player_id);
                        flag = false;
                        //replay.append(state);
                        break;
                    }
                    if (fishs[pos]->id == id){  //  断言成功
                        info_msg("正确。\n");
                        fishs[pos]->is_expose = true;
                        if (fishs[pos]->id == 12) (dynamic_cast<Imitator*>(fishs[pos]))->avatar->is_expose = true;
                        // fishs[pos]->atk_debuff(0.8);
                        game.players[player_id ^ 1].fight_fish.hp_debuff(50);
                        last_assert["AssertPos"] = pos;
                        last_assert["AssertContent"] = id;
                        last_assert["AssertResult"] = true;
                        game.players[0].fight_fish.update_state();
                        game.players[1].fight_fish.update_state();
                    }
                    else{  //  断言失败
                        info_msg("失败。\n");
                        game.players[player_id].fight_fish.hp_debuff(0);
                        last_assert["AssertPos"] = pos;
                        last_assert["AssertContent"] = id;
                        last_assert["AssertResult"] = false;
                    }
                    state["operation"].append(op);
                    replay.append(state);
                    game.players[0].fight_fish.update_state();
                    game.players[1].fight_fish.update_state();
                    break;
                }
                debug_msg("read assert end");
            }
            else if (game.gamestate == Game::ACTION){
                debug_msg("Action Start! (Player "+std::to_string(player_id)+")");
                if (!action_msg_checker(content, player_id)){      //玩家行动信息不合规范
                    flag = false;
                    game.errorai = (1 << player_id);
                    //replay.append(state);
                    break;
                }
                auto my_fishs = game.players[player_id].fight_fish.get_fishs();
                auto enemy_fishs = game.players[player_id ^ 1].fight_fish.get_fishs();
                int type = content["Type"].asBool();
                int my_pos = content["MyPos"].asInt();
                //last_action = {type, my_pos};
                op["Action"] = "Action";
                op["ID"] = player_id;
                op["Type"] = type;
                op["MyPos"] = my_pos;
                debug_msg("Action Middle! (type = "+std::to_string(type)+")");
                game.players[0].fight_fish.another = &game.players[1].fight_fish;
                game.players[1].fight_fish.another = &game.players[0].fight_fish;
                game.players[0].fight_fish.timestamp = game.players[1].fight_fish.timestamp = 0;
                if(type == 0){
                    int enemy_pos = content["EnemyPos"].asInt();
                    if (my_fishs[my_pos]->state == Fish::DEAD || enemy_fishs[enemy_pos]->state == Fish::DEAD){
                        game.errorai = (1 << player_id);
                        flag = false;
                        Json::Value error_info;
                        error_info["player"] = player_id;
                        error_info["type"] = ErrorType::action_with_dead_fish;
                        state["errors"].append(error_info);
                        //replay.append(state);
                        break;
                    }
                    op["EnemyPos"] = enemy_pos;
                    debug_msg(std::to_string(my_fishs[my_pos]->id) + " normal attack " + std::to_string(enemy_fishs[enemy_pos]->id));
                    debug_msg("======= Normal Attack Before =======");
                    action_info = my_fishs[my_pos]->attack(enemy_fishs[enemy_pos]);
                    //my_fishs[my_pos]->attack(enemy_fishs[enemy_pos]);
                    debug_msg("======= Normal Attack After =======");
                    if (action_info){
                        debug_msg("action is valid");
                        acted = true;           //平A合法
                        (*action_info)["ActionFish"] = my_pos;
                        last_action[player_id] = *action_info;
                    }
                    else
                        acted = false;          //平A不合法
                    info_msg("编号为 "+std::to_string(my_pos)+" 的鱼对敌方编号为 "+std::to_string(enemy_pos)+" 的鱼发动攻击，"+"造成 "+std::to_string(my_fishs[my_pos]->atk)+" 点伤害");
                }
                else{
                    debug_msg("======= Skill Before =======");
                    op["MyList"] = content["MyList"];
                    op["EnemyList"] = content["EnemyList"];
                    auto to_vector = [](const Json::Value& json,
                                        const std::vector<Fish*>& fishs)
                        -> std::vector<Fish*> {
                        std::vector<Fish*> ret(json.size());
                        if (json.size() == 0){
                            ret.resize(0);
                            return ret;
                        }
                        std::vector<int> position(json.size());
                        for (int i = 0; i < (int)json.size(); ++i)
                            position[i] = json[i].asInt();
                        std::sort(position.begin(), position.end());
                        for (int i = 0; i < (int)json.size(); ++i)
                            ret[i] = fishs[position[i]];
                            //ret[i] = fishs[json[i].asInt()];
                        return ret;
                    };
                    auto my_list = to_vector(content["MyList"], my_fishs);
                    auto enemy_list = to_vector(content["EnemyList"], enemy_fishs);

                    debug_msg(std::to_string(my_fishs[my_pos]->id) + " skill");
                    debug_msg("My_list:");
                    for(auto i:my_list)
                        debug_msg(std::to_string(i->id));
                    debug_msg("Enemy_list:");
                    for(auto i:enemy_list)
                        debug_msg(std::to_string(i->id));

                    bool has_death = false;
                    if (my_fishs[my_pos]->state == Fish::DEAD) has_death = true;
                    for (auto i:my_list){
                        if (i->state == Fish::DEAD){
                            has_death = true;
                            break;
                        }
                    }
                    for (auto i:enemy_list){
                        if (i->state == Fish::DEAD){
                            has_death = true;
                            break;
                        }
                    }
                    if (has_death){
                        game.errorai = (1 << player_id);
                        flag = false;
                        Json::Value error_info;
                        error_info["player"] = player_id;
                        error_info["type"] = ErrorType::action_with_dead_fish;
                        state["errors"].append(error_info);
                        //replay.append(state);
                        break;
                    }
                    //acted = my_fishs[my_pos]->skill(std::move(my_list), std::move(enemy_list));
                    debug_msg("action info before");
                    debug_msg((my_fishs[my_pos]->id == Fish::sunfish)?"fanbao!":"???");
                    action_info = my_fishs[my_pos]->skill(std::move(my_list), std::move(enemy_list));
                    debug_msg("action info after");
                    if (action_info){
                        debug_msg("action info valid");
                        acted = true;      //技能合法
                        (*action_info)["ActionFish"] = my_pos;
                        last_action[player_id] = *action_info;
                    }  
                    else
                        acted = false;     //技能不合法
                    info_msg("编号为 "+std::to_string(my_pos)+" 的鱼发动了主动技能");
                    debug_msg("acted = " + std::to_string(acted ? 1 : 0));
                    debug_msg("======= Skill After =======");
                }
                game.players[0].fight_fish.update_state();
                game.players[1].fight_fish.update_state();
                if (!acted){       //玩家行动不符合规则
                    game.errorai = (1 << player_id);
                    flag = false;
                    Json::Value error_info;
                    error_info["player"] = player_id;
                    error_info["type"] = ErrorType::action_rules_error;
                    error_info["actionfish"] = my_fishs[my_pos]->id;
                    error_info["action_rules_error_type"] = my_fishs[my_pos]->error_type;
                    state["errors"].append(error_info);
                    //replay.append(state);
                }
                debug_msg("Action End!\n");

                auto fishs_0 = game.players[0].fight_fish.get_fishs();
                auto fishs_1 = game.players[1].fight_fish.get_fishs();
                debug_msg("Status:");
                debug_msg("Player 0:");
                for(auto i:fishs_0) {
                    std::string s="Fish ";
                    s = s + std::to_string(i->id);
                    s = s + ": hp " + std::to_string(i->hp) + "/" + std::to_string(i->max_hp) + ", ";
                    s = s + "atk " + std::to_string(i->atk) + ", ";
                    s = s + "buff " + std::to_string(i->buff);
                    debug_msg(s);
                }
                debug_msg("Player 1:");
                for(auto i:fishs_1) {
                    std::string s="Fish ";
                    s = s + std::to_string(i->id);
                    s = s + ": hp " + std::to_string(i->hp) + "/" + std::to_string(i->max_hp) + ", ";
                    s = s + "atk " + std::to_string(i->atk) + ", ";
                    s = s + "buff " + std::to_string(i->buff);
                    debug_msg(s);
                }

                if (acted){
                    auto info = *action_info;
                    action_info_modifier(info, state["cur_turn"].asInt());
                    op["ActionInfo"] = info;
                    state["operation"].append(op);
                    replay.append(state);
                }
                break;
            }
        }
        debug_msg("read end");
        if (!flag){
            replay.append(state);
            if (game.errorai == 3) game.cur_turn = -1;
            else if (game.errorai == 2) game.cur_turn = 1;
            else game.cur_turn = 0;
        }
        return flag; // success
    }

    /*
        保存当前场面，即 game.to_json()     //(szh:以及该回合玩家进行的操作)
        文件格式: json
        文件样例:
                {
                    "winner": winner,
                    "players": [
                        {
                            "id": id,
                            "type": type,
                            "my_fish": [ // my_fish
                                {
                                    "hp": hp,
                                    "atk": atk,
                                    "def": def,
                                    "type": type,
                                    "id": id,
                                    "state": state,
                                    "is_expose": is_expose
                                }
                                fish2,
                                fish3,
                                ...
                            ],
                            "fight_fish": fight_fish
                        },
                        player2,
                        player3,
                        ...
                    ],
                    "state": state,
                    "gamestate": gamestate,
                    "cur_turn": cur_turn,
                    "over": over,
                    "cnt": cnt,
                    "score": score,
                    "rounds": rounds
                    "operation":[
                        {
                            "ID":player_id,
                            ...
                        },
                        {
                            "ID":player_id,
                            ...
                        }
                    ]
                },
    */
    void save_json(){
        state = game.to_json();
        // replay.append(state); state还要记录具体操作，应该在read_json中把state加入replay
    }

    /*
    游戏结束，向judger传输结束信息，并完成replay保存
    1. 传给judger結束信息
        通信格式: json
        通信内容: 游戏结束标志符
                 选手id列表
                 按照上面的列表依次给出选手的天梯分数增
        通信样例:
                {
                    "state": -1,
                    "end_info":
                    {
                        "0": 100,
                        "1": 0,
                    }
                }
    2. 录播文件保存結局
            {
                'winner': '0',
                '0': 100,
                '1': 0,
            }
    */
    void end(){
        Json::Value end_replay;
        Json::Value root;
        root["state"] = -1;
        Json::Value end_info;
        info_msg("游戏结束，玩家 "+std::to_string((game.score < 0))+" 胜利！\n");
        if (game.score > 0){
            end_info[std::to_string(0)] = 100;
            end_info[std::to_string(1)] = 0;
        }
        else if (game.score < 0){
            end_info[std::to_string(0)] = 0;
            end_info[std::to_string(1)] = 100;
        }
        else{
            end_info[std::to_string(0)] = 0;
            end_info[std::to_string(1)] = 0;
        }
        replay.append(end_replay);
        fout << replay.toStyledString() << "\n\n\n";
        fout.close();
        root["end_info"] = end_info.toStyledString();
        writer->write(root, &os);
        Msg = os.str();
        os.str("");  // 清空串流
        sendMsg(Msg);
        Msg = "";
    }

    //  游戏回合制运行
    void run(){
        debug_msg("logic run");
        //srand(time(0));
        while (!game.over){
            debug_msg("save start");
            save_json();  // 保存回合至录播文件
            debug_msg("save end");
            write_json();  // 发送消息给judger
            if(!read_json()){ // 接收judger的讯息
                invalid_operation();
                continue;
            }
            // xwb(2021.1.21): 仅确定先后手，不产生新的回合
            if (game.gamestate == Game::READY) {
                if (game.last_winner == -1){
                    //game.first_mover = rand() % 2;
                    game.first_mover = 0;
                }
                else {
                    game.first_mover = game.last_winner ^ 1;
                    ++game.state; // xwb(2021.1.21): 马上进行的 PICK 回合
                }
                game.cur_turn = game.first_mover;
                game.gamestate = Game::PICK;
                continue;
            }
           game.state ++;
           if(game.state - game.last_round_state > Game::STATE_LIMIT){
               state_limit_exceeded();
               continue;
           }
           if (game.gamestate == Game::PICK){ // xwb(2021.1.21): 分为两回合
                if (game.cur_turn != game.first_mover) {
                    game.gamestate = Game::ASSERT;
                }
                game.cur_turn ^= 1;
           }
           else if (game.gamestate == Game::ASSERT){
               bool flag = false;
               int winner;
               for(int i = 0; i < 2; i ++){
                   if (game.players[i].fight_fish.is_all_dead()){
                       if (game.players[i ^ 1].fight_fish.is_all_dead()){
                           winner = game.cur_turn;
                           flag = true;
                           break;
                       }
                       info_msg("\n\n第 "+std::to_string(game.rounds)+" 轮结束，玩家 ");
                       info_msg(std::to_string((i^1))+" 胜利！\n");
                       info_msg("=========================\n\n");
                       winner = i ^ 1;
                       /*if (i == 0) game.score --;
                       else game.score ++;
                       if (game.rounds == 3) game.over = true;*/
                       flag = true;
                       break;
                   }
               }
               if (flag){
                   round_end(winner, RoundEndFlag::NormalAssert);
               }
               else{
                   game.gamestate = Game::ACTION;
               }
           }
           else if (game.gamestate == Game::ACTION){
               bool flag = false;
               int winner;
               for(int i = 0; i < 2; i ++){
                   if (game.players[i].fight_fish.is_all_dead()){
                       if (game.players[i ^ 1].fight_fish.is_all_dead()){
                           winner = game.cur_turn;
                           flag = true;
                           break;
                       }
                       info_msg("\n\n第 " + std::to_string(game.rounds) + " 轮结束，玩家 ");
                       info_msg(std::to_string((i ^ 1)) + " 胜利！\n");
                       info_msg("=========================\n\n");
                       winner = i ^ 1;
                       flag = true;
                       break;
                   }
               }
               if (flag){
                   round_end(winner, RoundEndFlag::NormalAction);
               }
               else{
                   game.gamestate = Game::ASSERT;
                   game.cur_turn ^= 1;
               }
           }
        }
        end();  // 返回局面结束信息
    }

    // 处理 AI 非法操作的情形
    void invalid_operation() {
        if(game.cur_turn == 0) game.score = -3;
        else if (game.cur_turn == 1) game.score = 3;
        else game.score = 0;
        bool winner = game.cur_turn ^ 1;
        Json::Value root;
        root["state"] = game.state;
        bool flag = false;
        for (int j=0;j<2;j++){
            if (game.players[j].type != 2) continue;
            flag = true;
            root["listen"].append(j);
            root["player"].append(j);
            Json::Value player;
            player["Action"] = "EarlyFinish";
            player["Result"] = (winner==j ? "Win" : "Lose");
            root["content"].append(remove_enter(player.toStyledString()));
        }
        if (flag) {
            writer->write(root, &os);
            Msg = os.str();
            os.str(""); // 清空串流
		    data_msg("MSG @ 218: " + Msg + "\n");
            sendMsg(Msg);
            Msg = "";
        }
        end();
    }

    // 处理当前轮回合数超过限制的情形
    void state_limit_exceeded() {
        auto winner = [this]() -> int {
            auto compare = [this](int (FishSet::*pf)()const) ->int {
                int x0 = (game.players[0].fight_fish.*pf)();
                int x1 = (game.players[1].fight_fish.*pf)();
                if(x0 == x1) return -1;
                else return x1 > x0;
            };
            int ret;
            ret = compare(&FishSet::living_fish_count); // 活鱼多者胜
            if(ret != -1) return ret;
            ret = compare(&FishSet::hp_sum); // 总血量大者胜
            if(ret != -1) return ret;
            ret = compare(&FishSet::hp_max); // 最大血量大者胜
            if(ret != -1) return ret;
            return game.first_mover ^ 1; // 后手胜
        };
        if (game.gamestate == Game::ASSERT)
            round_end(winner(), RoundEndFlag::SLEAssert);
        else
            round_end(winner(), RoundEndFlag::SLEAction);
    }

    // 当前轮结束
    void round_end(int winner, RoundEndFlag flag) {
        game.last_round_state = game.state;
        game.last_winner = winner;
        if (winner == 0) game.score ++;
        else game.score --;
        game.rounds ++;
        if (game.rounds == 1) game.over = true;
        game.gamestate = Game::READY; // xwb(2021.1.21): 新一轮的先后手
        state = game.to_json();
        game.players[0].fight_fish.to_dead();
        game.players[1].fight_fish.to_dead();
        Json::Value root;
        root["state"] = game.state;
        //root["isStateLimiteExceed"] = (flag == RoundEndFlag::SLE);
        for (int j=0;j<2;j++){
            root["listen"].append(j);
            root["player"].append(j);
            Json::Value player;
            player["Action"] = "Finish";
            player["Result"] = (winner==j ? "Win" : "Lose");
            player["isStateLimitExceed"] = ((flag == RoundEndFlag::SLEAssert) || (flag == RoundEndFlag::SLEAction));
            if (flag == RoundEndFlag::NormalAssert || flag == RoundEndFlag::SLEAssert){    //断言阶段结束
                if (j == game.cur_turn){
                    player["MyAssert"] = last_assert;
                }
                else{
                    player["EnemyAssert"] = last_assert;
                    action_info_modifier(last_action[j], j);
                    player["EnemyAction"] = Json::nullValue;
                    player["MyAction"] = last_action[j];
                }
            }
            else if (flag == RoundEndFlag::NormalAction || flag == RoundEndFlag::SLEAction){    //行动阶段结束
                if (j == game.cur_turn){
                    action_info_modifier(last_action[j],j);
                    player["MyAction"] = last_action[j];
                }
                else{
                    player["EnemyAssert"] = last_assert;
                    for (int k=0;k<2;k++){
                        action_info_modifier(last_action[k],j);
                    }
                    player["EnemyAction"] = last_action[j ^ 1];
                    player["MyAction"] = last_action[j];
                }
            }
            root["content"].append(remove_enter(player.toStyledString()));
        }
        writer->write(root, &os);
        Msg = os.str();
        os.str(""); // 清空串流
        //fprintf(f, "MSG @ 218: %s\n", Msg.c_str());
        data_msg("MSG @ 218: " + Msg + "\n");
        sendMsg(Msg);
        Msg = "";
        replay.append(state);
    }
};