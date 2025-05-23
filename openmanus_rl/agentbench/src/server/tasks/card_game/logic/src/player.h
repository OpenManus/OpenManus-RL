#pragma once
#include "fishset.h"
#include <vector>

class Player {
  private:
    int id;  //  玩家id

  public:
    int type = 1;  //  1 =>  AI, 2 => 播放器
    FishSet my_fish;  //  队伍里的所有剩余鱼
    FishSet fight_fish;  //  当前上场战斗的鱼
    Player(int _id);
    Player(int _id, FishSet _my_fish);
    Player(const Player &p);
    Json::Value to_json() const;
    int get_id();
    std::vector<Fish*> get_fishs() const;
    std::vector<Fish*> get_fight_fishs() const;
    int get_size();
    bool empty();
    void clear();
    void add(Fish* fish);
    void add(int id);  //  添加对应id的鱼
    void remove(int id);  //  移除对应id的鱼
    void to_war(int id);  //  编号为id的一只鱼上场战斗
};