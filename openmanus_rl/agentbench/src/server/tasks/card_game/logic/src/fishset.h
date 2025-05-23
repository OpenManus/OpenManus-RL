#pragma once
#include "fish.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <algorithm>

extern std::ofstream debuggerfs;
void debugmsg(std::string str);

class FishSet
{
private:
  std::vector<Fish *> fishs;

  public:
    int player_id;
    int timestamp;
    FishSet* another = nullptr;
    FishSet(int flag = 0);
    FishSet(const FishSet& base);  //  根据一个已知鱼集合创建，得到一个鱼编号集合和给定集合一致的鱼集合
    Json::Value to_json() const;
    std::vector<Fish*> get_fishs() const;
    int get_size();
    bool empty();
    void clear();
    void add(Fish* fish);
    void add(int id);  //  添加对应id的鱼
    void remove(int id);  //  移除对应id的鱼
    void to_fight();  //  所有鱼变为战斗状态
    void to_dead();  //  所有鱼变为阵亡状态
    void update_state();  //  将所有鱼从战斗状态更新到最新状态（可能是阵亡状态）
    // void hp_debuff(double rate);
    void hp_debuff(int dec);
    int count_live_fish();
    int living_fish_count() const; // 存活的鱼的个数
    int hp_sum() const; // 鱼的 hp 之和
    int hp_max() const; // 鱼的 hp 最大值
    bool is_all_dead();
    bool count(Fish* tar);
    int update_timestamp();

  std::optional<Json::Value> on_damaged(Fish *src, Fish *target, int dmg = -1);
  void set_fishset(FishSet *_ally, FishSet *_hostile);
  void start_turn();
  // void debug_msg(std::string str);
};