#pragma once
#include "jsoncpp/json/json.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <optional>

typedef Json::Value J;
typedef std::optional<Json::Value> opt;

class Fish;

Json::Value skill_to_json(std::string type, std::vector<Fish*> targets, int player, int value, bool is_skill);
Json::Value passive_to_json(std::string type, int source, int player, float value, int time);
Json::Value hit_to_json(int target, int player, int value, bool traceable, int time);
void add_json(J& a, J b);

class FishSet;

class Fish {
  public:
    // static const int HP_DEC = 50;

    enum FishType{
        kawaii,
        // attached      //被加印
    };

    //  TODO: 完成鱼的ID及派生类定义
    enum FishID{
        spray = 1,
        flame,
        eel,
        sunfish,
        blind,
        flyfish,
        turtle,
        octopus,
        whiteshark,
        goblinshark,
        // cowfish,
        // suckerfish,
        // Pufferfish,
        // Lion,
        // Carp,
        // Doctor,
        joker,
        imitator
    };

    enum FishState{
        READY,FIGHT,DEAD
    };

    enum FishBuff{
        SWELL = 1,    // 膨胀
        HEAL = 2,     // 生命低于10%时自动恢复15%血量 (1层)
        SHIELD = 4,   // 下次被攻击时减免70%伤害
        LIFEONHIT = 8,// 受击恢复20血
        DEFLECT = 16  // 下次被攻击后队友分摊
    };

    enum FishSkillType{
        AOE,

    };

    enum FishErrorType{
        friend_target_not_empty = 1,
        friend_target_empty,
        enemy_target_not_empty,
        enemy_target_empty,
        friend_target_too_much,
        enemy_target_too_much,
        infight_myself,
        buff_myself,
        enemy_target_hp_not_lowest,
        enemy_target_not_all
    };

    int hp;  //  生命值
    int max_hp; // max_hp
    int wound = 0; //累计受伤 % 200
    int special_usage = 0;//海龟、小丑鱼技能使用次数
    int shell = 3;// 海龟的盾
    int atk;  //  攻击力
    int def;  // 防御力
    int pos;  //  鱼的位置
    int id;  //  鱼的编号（是哪只鱼）
    int state = READY;  //  鱼的状态
    bool is_expose = false;  //  鱼的暴露状态
    int buff = 0; // 被动效果
    int error_type = -1;
    
    Fish* original = nullptr; // 如果它是被拟态鱼构造的，指向之
    // Fish* attach_target = nullptr;  // 吸附目标
    // int attach_type;                // 吸附目标是队友 (0) 或敌人 (1)


    FishSet* ally;
    FishSet* hostile;

    std::ofstream debugger;

    Fish(int _hp, int _atk, int _def, int _pos, int _id, FishSet* _ally=nullptr, FishSet* _hostile=nullptr);
    static bool is_valid_fish_type(int type);
    static bool is_valid_fish_id(int id);
    int update_timestamp();

    virtual int get_atk();
    virtual Json::Value to_json() const;

    virtual void set_fishset(FishSet* _ally, FishSet* _hostile);
    virtual std::optional<Json::Value> take_damage(Fish* src, int dmg);
    virtual int heal_hp(int num);

    virtual void add_buff(int buff);
    virtual void remove_buff(int buff);
    virtual bool check_buff(int _buff);
    virtual void hp_debuff(int dec);
    
    // not used
    // virtual void atk_debuff(double rate);

    virtual std::optional<Json::Value> attack(Fish *target);
    virtual bool skill_valid(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
    virtual std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);

    virtual std::optional<Json::Value> bonus_attack(std::vector<Fish*> target);
    virtual std::optional<Json::Value> explode(Fish* target);
    virtual void start_turn();

    virtual void debug_msg(std::string str);
};

// ---------------------------1--------------------------

class Spray : public Fish {
public:
  Spray() : Fish(400, 200, 0, Fish::kawaii, Fish::spray, nullptr, nullptr) {

  }
  // std::optional<Json::Value> take_damage(Fish* src, int dmg);
  using Fish::take_damage;
  opt special(Fish* self, std::vector<Fish*> enemy_list);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Flame :public Fish {
public:
  Flame() : Fish(400, 200, 0, Fish::kawaii, Fish::flame, nullptr, nullptr) {

  }
  using Fish::take_damage;
  opt special(std::vector<Fish*> target_list);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Eel : public Fish {
public:
  Eel() : Fish(400, 200, 0, Fish::kawaii, Fish::eel, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  opt special(Fish* self, std::vector<Fish*> target_list);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Sunfish : public Fish {
public:
  Sunfish() : Fish(400, 200, 0, Fish::kawaii, Fish::sunfish, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  //opt special(Fish* self, std::vector<Fish*> target_list);
  opt special(std::vector<Fish*> target_list);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Blind : public Fish {
public:
  Blind() : Fish(400, 100, 0, Fish::kawaii, Fish::blind, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  opt special(Fish* self, std::vector<Fish*> target_list);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Flyfish : public Fish {
public:
  Flyfish() : Fish(400, 100, 0, Fish::kawaii, Fish::flyfish, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  opt special(Fish* self, std::vector<Fish*> target_list);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

// ---------------------------7--------------------------

class Turtle : public Fish {
public:
  Turtle() : Fish(400, 100, 0, Fish::kawaii, Fish::turtle, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  opt special(Fish* self, Fish *target);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
  std::optional<Json::Value> bonus_attack(std::vector<Fish*> target);
};

class Octopus : public Fish {
public:
  Octopus() : Fish(400, 100, 0, Fish::kawaii, Fish::octopus, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  opt special(Fish* self, Fish* target);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Whiteshark : public Fish {
public:
  Whiteshark() : Fish(400, 100, 0, Fish::kawaii, Fish::whiteshark, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  opt special(Fish* self, std::pair<Fish*, int>& effect, Fish* target);
  // using Fish::start_turn;
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
};

class Goblinshark : public Fish {
public:
  Goblinshark() : Fish(400, 100, 0, Fish::kawaii, Fish::goblinshark, nullptr, nullptr) {

  }
  std::optional<Json::Value> take_damage(Fish* src, int dmg);
  int get_atk();
  // using Fish::take_damage;
  opt special(Fish* self, std::pair<Fish*, int>& effect, Fish* target);
  std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
  std::optional<Json::Value> explode(Fish* target);
};

class Joker : public Fish {
  public:
    Joker() : Fish(400, 100, 0, Fish::kawaii, Fish::joker, nullptr, nullptr) {

    }
    // Joker(int _hp, int _atk, int _def, int _type, FishSet* _ally, FishSet* _hostile):
    //   Fish(_hp, _atk, _def, _type, FishID::Joker, _ally, _hostile) {}
    std::optional<Json::Value> take_damage(Fish* src, int dmg);
    opt special(Fish* tar);
    std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);
    std::optional<Json::Value> bonus_attack(std::vector<Fish*> target);
};

// to do
class Imitator : public Fish {
  public:
    Imitator() : Fish(400, 100, 0, Fish::kawaii, Fish::imitator, nullptr, nullptr) {

    }
    // Imitator(int _hp, int _atk, int _def, int _type, FishSet* _ally, FishSet* _hostile):
    //   Fish(_hp, _atk, _def, _type, FishID::Imitator, _ally, _hostile) {}
    opt special(Fish* self, std::vector<Fish*> target, int type=0);
    std::optional<Json::Value> skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list);

    Fish* avatar;
    void imitate(int id);//拟态前要先赋值ally和hostile,否则都是nullptr!
    void sync();

    int heal_hp(int num);
    void add_buff(int buff);
    void remove_buff(int buff);
    bool check_buff(int _buff);
    void hp_debuff(int dec);
    int get_atk();
    //void atk_debuff(double rate);
    std::optional<Json::Value> attack(Fish *target);
    std::optional<Json::Value> take_damage(Fish* src, int dmg);
    std::optional<Json::Value> bonus_attack(std::vector<Fish*> target);
};
//  根据id创建鱼
Fish* create_fish(int id);
