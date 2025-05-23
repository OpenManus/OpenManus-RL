#include "fish.h"
#include "fishset.h"
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <random>

using std::nullopt;
using std::optional;

//std::default_random_engine eng(time(0));
std::random_device rd;
std::default_random_engine eng(rd());

double random(double l, double r) {
    std::uniform_real_distribution<float> distr(l, r);
    return distr(eng);
}

void Fish::debug_msg(std::string str){
    return;
    debugger.open("debug_info.txt", std::ios::out | std::ios::app);
    debugger << str << std::endl;
    debugger.close();
}

Json::Value skill_to_json(std::string type, std::vector<Fish*> targets, int player, int value, bool is_skill) {
    Json::Value json_skill, json_targets;
    json_skill["type"] = type;
    json_skill["isSkill"] = is_skill;
    for(auto i : targets) {
        Json::Value json_target;
        json_target["pos"] = i->pos;
        json_target["player"] = player;
        json_target["value"] = value;
        json_targets.append(json_target);
    }
    json_skill["targets"] = json_targets;
    return json_skill;
}

Json::Value passive_to_json(std::string type, int source, int player, float value, int time) {
    Json::Value json;
    json["type"] = type;
    json["source"] = source;
    json["player"] = player;
    json["value"] = value;
    json["time"] = time;
    return json;
}

Json::Value hit_to_json(int target, int player, int value, bool traceable, int time) {
    Json::Value json;
    json["target"] = target;
    json["player"] = player;
    json["value"] = value;
    json["traceable"] = traceable;
    json["time"] = time;
    return json;
}

void add_json(J& a, J b) {
    for(auto i : b["passive"]) {
        a["passive"].append(i);
    }
    for(auto i : b["hit"]) {
        a["hit"].append(i);
    }
}

Fish::Fish(int _hp, int _atk, int _def, int _pos, int _id, FishSet* _ally, FishSet* _hostile):
    hp(_hp), atk(_atk), def(_def), pos(_pos), id(_id), ally(_ally), hostile(_hostile) {
        max_hp = hp;
}

bool Fish::is_valid_fish_type(int type){
    // return type >= 0 && type <= 2;
    return false;
}

//  TODO: 完成判断ID合法
bool Fish::is_valid_fish_id(int id){
    if (id < 1 || id > 12) return false;
    else return true;
}

int Fish::update_timestamp(){
    ally->timestamp ++;
    hostile->timestamp ++;
    return ally->timestamp;
}

int Fish::get_atk(){
    return atk;
}

/*
    按照成员变量定义顺序转成 JSON
    样例：
        {
            "hp": hp,
            "atk": atk,
            "def": def,
            "type": type,
            "id": id,
            "state": state,
            "is_expose": is_expose
        }
*/
Json::Value Fish::to_json() const
{
    Json::Value json;
    json["hp"] = hp;
    int now_atk = atk;
    int now_id = id;
    if (auto imitate = dynamic_cast<const Imitator *>(this))
        if (imitate->avatar)
            now_id = imitate->avatar->id;
    if (now_id == 10 && hp * 5 < max_hp) now_atk += 15;
    json["atk"] = now_atk;
    json["def"] = def;
    json["pos"] = pos;
    if (ally != nullptr)
        json["player"] = ally->player_id;
    json["id"] = id;
    json["state"] = state;
    json["is_expose"] = is_expose;
    if (auto imitate = dynamic_cast<const Imitator *>(this))
        if (imitate->avatar)
            json["imitate"] = imitate->avatar->id;
    return json;
}

int Fish::heal_hp(int num){
    int hurt = max_hp - hp;
    if(hp + num > max_hp) {
        hp = max_hp;
    } else {
        hp += num;
        hurt = num;
    }
    return hurt;
}

void Fish::add_buff(int _buff){
    buff |= _buff;
}

void Fish::remove_buff(int _buff){
    buff &= (-1)^_buff;
}

bool Fish::check_buff(int _buff){
    return (buff & _buff) > 0;
}

void Fish::hp_debuff(int dec){
    // hp = int(hp * rate);
    hp = (hp >= dec ? hp - dec : 0);
    //if (hp <= 0) state = DEAD;
    //在行动结束后再处理DEAD！
}

/*
 * not used
void Fish::atk_debuff(double rate){
    atk = int(atk * rate);
    assert(atk > 0);
}
*/

/*
 * 小丑鱼被动：行动回合恢复自身 5%血量
 */
void Fish::start_turn() {
//     remove_buff(Fish::SWELL);
//     // buff &= (-1) ^ Fish::FishBuff::SWELL;
//     // if(id == FishID::Carp || id == FishID::Doctor){
//     //     add_buff(Fish::HEAL);
//     // }
}

/*
对于所有的 take_damage() 函数：
若 src == nullptr, 则伤害无来源，不触发 (反伤, 平摊伤害) 类技能
(被反伤, 被平摊伤害) 视为无来源伤害
(牛角鱼, 鱼印鱼) 吸附在队友身上时队友受到攻击：队友受到的 60% 伤害有来源，(牛角鱼, 鱼印鱼) 受到的 40% 伤害无来源

代码中来自队友的伤害被视为有来源, 可触发 (反伤, 平摊伤害)
*/

opt Fish::take_damage(Fish* src, int dmg){
    if(state == FishState::DEAD) return nullopt;
    debug_msg("Fish take damage");
    J json;
    int d = dmg;
    int live_fishs = ally->count_live_fish();
    if(check_buff(Fish::SHIELD) && src) {
        remove_buff(Fish::SHIELD);
        d *= 0.3;
        json["passive"].append(passive_to_json("reduce", (original ? original : this)->pos, ally->player_id, 0.3, update_timestamp()));
    }
    if(check_buff(Fish::DEFLECT) && src && live_fishs > 1) {
        remove_buff(Fish::DEFLECT);
        int cnt = 0;
        auto e = ally->get_fishs();
        for(auto i: e)
            if(i != this && i != original && i->state != FishState::DEAD)
                ++cnt;
        // if (cnt > 0) d /= cnt;
        json["passive"].append(passive_to_json("deflect", (original ? original : this)->pos, ally->player_id, d, update_timestamp()));
        int dd = d * 0.3;
        for(auto i:e)
            if(i->state != Fish::DEAD && i != this && i != original) {
                add_json(json, (i->take_damage(nullptr, dd / cnt)).value());
            }
        d = d * 0.7;
    }
    // int tmp = d;
    // auto e = ally->get_fishs();
    // for(auto i: e)
    //     if(i->attach_target == this && i->attach_type == 0){
    //         i->take_damage(nullptr, tmp*0.4);
    //         d -= tmp*0.4;
    //     }
    if(src) {
        json["hit"].append(hit_to_json((original ? original : this)->pos, ally->player_id, d, true, update_timestamp()));
    } else {
        json["hit"].append(hit_to_json((original ? original : this)->pos, ally->player_id, d, false, update_timestamp()));
    }
    hp -= d;
    wound += d;
    if(hp <= 0){
        debug_msg("Fish take damage dead");
        //state = DEAD;
        //在行动结束后处理DEAD!
        return json;
    }
    /*if(hp <= int(0.1*max_hp) && check_buff(FishBuff::HEAL)) {
        remove_buff(FishBuff::HEAL);
        json["passive"].append(passive_to_json("heal", (original ? original : this)->pos, ally->player_id, heal_hp(0.15*max_hp)));
    }*/
    if(check_buff(FishBuff::LIFEONHIT) && src) {
        remove_buff(FishBuff::LIFEONHIT);
        json["passive"].append(passive_to_json("heal", (original ? original : this)->pos, ally->player_id, heal_hp(20), update_timestamp()));
    }
    /*if(check_buff(FishBuff::SWELL) && src) {
        add_json(json, (src->take_damage(nullptr, dmg)).value());
        json["passive"].append(passive_to_json("counter", (original ? original : this)->pos, ally->player_id, 0));
    }*/
    return json;
}

void Fish::set_fishset(FishSet* _ally, FishSet* _hostile) {
    ally = _ally, hostile = _hostile;
}

/*
加布林鲨鱼和牛角鱼的被动
*/
std::optional<Json::Value> Fish::attack(Fish* target) {
    //int dmg = atk / 2;
    int dmg;
    if((id == Fish::goblinshark) && hp * 5 < max_hp) dmg = (atk + 15) / 2;
    else dmg = atk / 2;
    assert(hostile != nullptr);
    Fish* src = original ? original : this;

    J json = hostile->on_damaged(src, target, dmg).value();
    std::vector<Fish*> targets;
    targets.clear();
    targets.push_back(target);
    json["skill"] = skill_to_json("normalattack", targets, hostile->player_id, dmg, 0);
    return json;
}

std::optional<Json::Value> Fish::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    return nullopt;
}

std::optional<Json::Value> Fish::bonus_attack(std::vector<Fish*> target){
    return nullopt;
}

std::optional<Json::Value> Fish::explode(Fish* target){
    return nullopt;
}

bool Fish::skill_valid(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    int n = my_list.size(), m = enemy_list.size();
    if (state == FishState::DEAD) return 0;
    for(auto i:my_list) {
        //if(!ally -> count(i)) return 0;
        if(i -> state == FishState::DEAD) return 0;
    }
    for(auto i:enemy_list) {
        //if(!hostile -> count(i)) return 0;
        if(i -> state == FishState::DEAD) return 0;
    }
    for(int i=0; i<n; i++)
        for(int j=i+1; j<n; j++)
            if(my_list[i]->id == my_list[j]->id) return 0;
    for(int i=0; i<m; i++)
        for(int j=i+1; j<m; j++)
            if(enemy_list[i]->id == enemy_list[j]->id) return 0;
    return 1;
}

/*
Fish 1-12 and fish 13-18 were implemented in two slightly different ways.
to be unified
*/

/* ------------------------1-----------------------
 For fish 1-12: take_damage() always returns 1
*/

/*
 * 被动：队友被攻击后若其生命值少于30%，对攻击队友的敌人造成30的伤害
 * special: 对敌方多名目标发动攻击每个造成35的aoe伤害
 */
// int Spray::take_damage(Fish *src, int dmg)
// {
//     if(hp*2 < max_hp) add_buff(Fish::SWELL);
//     if(!Fish::take_damage(src, dmg)) return 0;
//     if(hp*2 < max_hp) add_buff(Fish::SWELL);
//     return 1;
// }

opt Spray::special(Fish* self, std::vector<Fish*> target_list) {
    J json;
    for (auto enemy : target_list)
    {
        add_json(json, self->hostile->on_damaged(self, enemy, 70).value());
    }
    return json;
}

opt Spray::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    //if(my_list.size() != 0 || enemy_list.size() == 0) return nullopt;
    if (my_list.size() != 0){
        error_type = Fish::friend_target_not_empty;
        return nullopt;
    }
    if (enemy_list.size() == 0){
        error_type = Fish::enemy_target_empty;
        return nullopt;
    }
    if (enemy_list.size() != hostile->count_live_fish()){
        error_type = Fish::enemy_target_not_all;
        return nullopt;
    }
    /*for(auto fish : enemy_list)
        if (fish->state == FishState::DEAD) return nullopt;*/

    Fish* src = original ? original : this;
    J json = Spray::special(src, enemy_list).value();
    json["skill"] = skill_to_json("aoe", enemy_list, hostile->player_id, 70, 1);
    /*for(auto enemy : enemy_list) {
        int id = enemy->pos;
        for(auto hits : json["hit"]) {
            if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
                for(J& targets : json["skill"]["targets"]) {
                    if(targets["pos"].asInt64() == id) {
                        targets["value"] = hits["value"].asInt64();
                        break;
                    }
                }
            }
        }
    }*/
    return json;
}

// ------------------------2-----------------------
/*
 * 被动：（在fishset中实现）队友被攻击后若其生命值少于 30%,对攻击队友的敌人造成一定伤害
 * 主动：let an ally take damage to gain attack boost
 */
opt Flame::special(std::vector<Fish*> target_list)
{
    Fish* target = target_list[0];
    J tmp = target->take_damage(nullptr, 75).value();
    atk += 140;
    return tmp;
}

optional<Json::Value> Flame::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    //if(my_list.size() != 1 || enemy_list.size() != 0) return nullopt;
    if (my_list.size() != 1){
        if (my_list.size() == 0) error_type = Fish::friend_target_empty;
        else error_type = Fish::friend_target_too_much;
        return nullopt;
    }
    if (enemy_list.size() != 0){
        error_type = Fish::enemy_target_not_empty;
        return nullopt;
    }
    //if(my_list[0]->id == this->id) return nullopt;
    //if(my_list[0]->id == (original?original:this)->id) return nullopt;
    if (my_list[0]->id == (original?original:this)->id){
        error_type = Fish::infight_myself;
        return nullopt;
    }
    //if (my_list[0]->state == FishState::DEAD) return nullopt;
    
    J json = Flame::special(my_list).value();

    std::vector<Fish*> target;
    target.clear();
    target.push_back(my_list[0]);

    json["skill"] = skill_to_json("infight", target, ally->player_id, 75, 1);
    return json;
}

// ------------------------3-----------------------
/*
 *被动：share damage with allies and gain attack boost when taking damage more than 5% max hp
 *主动：area attack
*/
/*
int Eel::take_damage(Fish *src, int dmg)
{
    int num_friend = 0;
    for (auto allies : ally->get_fishs())
    {
        if (allies->state != Fish::DEAD)
        {
            ++num_friend;
        }
    }
    dmg /= num_friend;
    for (auto allies : ally->get_fishs())
    {
        if (allies->state != Fish::DEAD)
        {
            ally->on_damaged(src, allies, dmg);
        }
    }
    hp -= dmg;
    if (hp <= 0)
    {
        state = Fish::DEAD;
        return 0;
    }
    if (dmg >= max_hp / 20)
    {
        atk += 10;
    }
    return 1;
}*/

opt Eel::take_damage(Fish *src, int dmg)
{
    J json;
    int hit = dmg;
    int live_fishs = ally->count_live_fish();
    if(src && live_fishs > 1){
        hit = dmg * 0.7;
        dmg = dmg * 0.3;
        int cnt = 0;
        auto e = ally->get_fishs();
        for(auto i: e)
            if(i != this && i != original && i->state != FishState::DEAD)
                ++cnt;
        if (cnt > 0) dmg /= cnt;
        json["passive"].append(passive_to_json("deflect", (original?original:this)->pos, ally->player_id, hit, update_timestamp()));
        for(auto i:e)
            if(i->state != FishState::DEAD && i != this && i != original) {
                add_json(json, i->take_damage(nullptr, dmg).value());
            }
        add_json(json, Fish::take_damage(src, hit).value());
                
    }else{
        if (src)
            json = Fish::take_damage(src, dmg).value();
        else
            json = Fish::take_damage(nullptr, dmg).value();
    }
    // wound += hit;
    // 在Fish::take_damage中结算伤害
    if(wound >= 200) {
        atk += (40 * (wound / 200));
        wound %= 200;
    }
    //if (hit >= 200) atk += 20;
    return json;
}

opt Eel::special(Fish* self, std::vector<Fish*> target_list) {
    int now_atk = atk;
    J json;
    for (auto enemy : target_list)
    {
        //add_json(json, self->hostile->on_damaged(self, enemy, 35).value());
        add_json(json, self->hostile->on_damaged(self, enemy, now_atk * 0.35).value());
    }
    return json;
}

optional<Json::Value> Eel::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    /*if(my_list.size() != 0 || enemy_list.size() == 0) return nullopt;
    for(auto fish : enemy_list)
        if (fish->state == FishState::DEAD) return nullopt;*/
    if (my_list.size() != 0){
        error_type = Fish::friend_target_not_empty;
        return nullopt;
    }
    if (enemy_list.size() == 0){
        error_type = Fish::enemy_target_empty;
        return nullopt;
    }
    if (enemy_list.size() != hostile->count_live_fish()){
        error_type = Fish::enemy_target_not_all;
        return nullopt;
    }

    Fish* src = original ? original : this;

    int now_atk = atk * 0.35;
    J json = Eel::special(src, enemy_list).value();
    json["skill"] = skill_to_json("aoe", enemy_list, hostile->player_id, now_atk, 1);
    /*for(auto enemy : enemy_list) {
        int id = enemy->pos;
        for(auto hits : json["hit"]) {
            if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
                for(J& targets : json["skill"]["targets"]) {
                    if(targets["pos"].asInt64() == id) {
                        targets["value"] = hits["value"].asInt64();
                        break;
                    }
                }
            }
        }
    }*/
    return json;
}

// ------------------------4-----------------------
/*
 *被动：share damage with allies and gain attack boost when taking damage more than 5% max hp
 *主动：let an ally take damage to gain attack boost
*/
opt Sunfish::take_damage(Fish *src, int dmg)
{
    debug_msg("Sunfish take damage start");
    int hit = dmg;
    int live_fishs = ally->count_live_fish();
    J json;
    if(src && live_fishs > 1){
        hit = dmg * 0.7;
        dmg = dmg * 0.3;
        int cnt = 0;
        auto e = ally->get_fishs();
        for(auto i: e)
            if(i != this && i != original && i->state != FishState::DEAD)
                ++cnt;
        if (cnt > 0) dmg /= cnt;
        json["passive"].append(passive_to_json("deflect", (original?original:this)->pos, ally->player_id, hit, update_timestamp()));
        for(auto i:e)
            if(i->state != FishState::DEAD && i != this && i != original)
                add_json(json, i->take_damage(nullptr, dmg).value());
        add_json(json, Fish::take_damage(src, hit).value());
    }else{
        if (src)
            json = Fish::take_damage(src, dmg).value();
        else
            json = Fish::take_damage(nullptr, dmg).value();
    }
    // wound += hit;
    // 在Fish::take_damage中结算伤害
    if(wound >= 200) {
        atk += (40 * (wound / 200));
        wound %= 200;
    }
    //if (hit >= 200) atk += 20;
    debug_msg("Sunfish take damage end");
    return json;
}

opt Sunfish::special(std::vector<Fish*> target_list)
{
    Fish* target = target_list[0];
    J json = target->take_damage(nullptr, 75).value();
    atk += 140;
    return json;
}

optional<Json::Value> Sunfish::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    debug_msg("Sunfish skill start");
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    /*if(my_list.size() != 1 || enemy_list.size() != 0) return nullopt;
    debug_msg("Sunfish skill is valid");
    if(my_list[0]->id == (original?original:this)->id) return nullopt;
    debug_msg("Sunfish skill is valid2");
    // if(my_list[0] == this) return 0;
    if (my_list[0]->state == FishState::DEAD) return nullopt;
    debug_msg("Sunfish skill is valid3");*/
    if (my_list.size() != 1){
        if (my_list.size() == 0) error_type = Fish::friend_target_empty;
        else error_type = Fish::friend_target_too_much;
        return nullopt;
    }
    if (enemy_list.size() != 0){
        error_type = Fish::enemy_target_not_empty;
        return nullopt;
    }
    if (my_list[0]->id == (original?original:this)->id){
        error_type = Fish::infight_myself;
        return nullopt;
    }

    J json = Sunfish::special(my_list).value();
    
    std::vector<Fish*> target;
    target.clear();
    target.push_back(my_list[0]);
    json["skill"] = skill_to_json("infight", target, ally->player_id, 75, 1);
    return json;
}

// ------------------------5-----------------------
// 梭子鱼
/*
 *被动：have a chance of 30% to evade
 *主动：critically attack
*/
opt Blind::take_damage(Fish *src, int dmg)
{
    J json;
    //srand(time(0));
    //if (rand() % 10 >= 3)
    if (random(0, 1) > 0.3)
        json = Fish::take_damage(src, dmg).value();
    else {
        json["passive"].append(passive_to_json("reduce", (original?original:this)->pos, ally->player_id, 0, update_timestamp()));
        json["hit"].append(hit_to_json((original?original:this)->pos, ally->player_id, 0, src, update_timestamp()));
    }
    return json;
}

opt Blind::special(Fish* self, std::vector<Fish*> target_list)
{
    Fish* target = target_list[0];
    return self->hostile->on_damaged(self, target, 120).value();
}

optional<Json::Value> Blind::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    /*if(my_list.size() != 0 || enemy_list.size() != 1) return nullopt;
    if (enemy_list[0]->state == FishState::DEAD) return nullopt;*/
    if (my_list.size() != 0){
        error_type = Fish::friend_target_not_empty;
        return nullopt;
    }
    if (enemy_list.size() != 1){
        if (enemy_list.size() == 0) error_type = Fish::enemy_target_empty;
        else error_type = Fish::enemy_target_too_much;
        return nullopt;
    }

    Fish* src = original ? original : this;
    J json = Blind::special(src, enemy_list).value();
    
    std::vector<Fish*> target;
    target.clear();
    target.push_back(enemy_list[0]);
    json["skill"] = skill_to_json("crit", target, hostile->player_id, 120, 1);
    // for(auto enemy : target) {
    //     int id = enemy->pos;
    //     for(auto hits : json["hit"]) {
    //         if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
    //             for(J& targets : json["skill"]["targets"]) {
    //                 if(targets["pos"].asInt64() == id) {
    //                     targets["value"] = hits["value"].asInt64();
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
    return json;
}

// ------------------------6-----------------------
//蝠鲼
/*
 *被动：have a chance of 30% to evade
 *主动：apply shield to an ally
*/
opt Flyfish::take_damage(Fish *src, int dmg)
{
    J json;
    //srand(time(0));
    //if (rand() % 10 >= 3)
    if (random(0, 1) > 0.3)
        json = Fish::take_damage(src, dmg).value();
    else {
        json["passive"].append(passive_to_json("reduce", (original?original:this)->pos, ally->player_id, 0, update_timestamp()));
        json["hit"].append(hit_to_json((original?original:this)->pos, ally->player_id, 0, src, update_timestamp()));
    }
    return json;
}

opt Flyfish::special(Fish* self, std::vector<Fish*> target_list)
{
    Fish* target = target_list[0];
    target->add_buff(FishBuff::SHIELD);
    self->atk += 20;
    // target->buff |= FishBuff::SHIELD;
    return J();
}

optional<Json::Value> Flyfish::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    /*if(my_list.size() != 1 || enemy_list.size() != 0) return nullopt;
    if (my_list[0]->state == FishState::DEAD) return nullopt;*/
    if (my_list.size() != 1){
        if (my_list.size() == 0) error_type = Fish::friend_target_empty;
        else error_type = Fish::friend_target_too_much;
        return nullopt;
    }
    if (enemy_list.size() != 0){
        error_type = Fish::enemy_target_not_empty;
        return nullopt;
    }
    
    J json = Flyfish::special(this, my_list).value();
    json["skill"] = skill_to_json("subtle", std::vector<Fish*>(), ally->player_id, 0, 1);
    return json;
}

//------------------------7-----------------------
// 海龟
opt Turtle::take_damage(Fish *src, int dmg)
{
    J json;
    if(shell > 0) {
        --shell;
        json["passive"].append(passive_to_json("reduce", (original?original:this)->pos, ally->player_id, 0, update_timestamp()));
        json["hit"].append(hit_to_json((original?original:this)->pos, ally->player_id, 0, src, update_timestamp()));
        return json;
    }
    //srand(time(0));
    //if (rand() % 10 >= 3)
    if (random(0, 1) > 0.3)
        json = Fish::take_damage(src, dmg).value();
    else {
        json["passive"].append(passive_to_json("reduce", (original?original:this)->pos, ally->player_id, 0, update_timestamp()));
        json["hit"].append(hit_to_json((original?original:this)->pos, ally->player_id, 0, src, update_timestamp()));
    }
    return json;
}

opt Turtle::special(Fish* self, Fish *target)
{
    target->add_buff(FishBuff::LIFEONHIT);
    ++special_usage;
    return J();
    // target->buff |= FishBuff::LIFEONHIT;
}

optional<Json::Value> Turtle::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    debug_msg("Turtle skill start");
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    //if(my_list.size() != 1 || special_usage < 3 && enemy_list.size() != 1) return nullopt;
    if (my_list.size() != 1){
        if (my_list.size() == 0) error_type = Fish::friend_target_empty;
        else error_type = Fish::friend_target_too_much;
        return nullopt;
    }
    if (special_usage < 3 && enemy_list.size() != 1){
        if (enemy_list.size() == 0) error_type = Fish::enemy_target_empty;
        else error_type = Fish::enemy_target_too_much;
        return nullopt;
    }
    if (special_usage >= 3 && enemy_list.size() != 0){
        error_type = Fish::enemy_target_not_empty;
        return nullopt;
    }
    if(my_list[0]->id == (original?original:this)->id){
        error_type = Fish::buff_myself;
        return nullopt;
    }
    /*if (my_list[0]->state == FishState::DEAD) return nullopt;
    if (special_usage < 3 && enemy_list[0]->state == FishState::DEAD) return nullopt;*/
    
    debug_msg("Turtle skill special before");
    J json = Turtle::special(this, my_list[0]).value();
    debug_msg("Turtle skill special after");
    json["skill"] = skill_to_json("subtle", std::vector<Fish*>(), ally->player_id, 0, 1);
    if(special_usage <= 3) {
        J bonus = bonus_attack(enemy_list).value();
        add_json(json, bonus);
        json["skill"] = bonus["skill"];
    }
    debug_msg("Turtle skill to json after");
    return json;
}

std::optional<Json::Value> Turtle::bonus_attack(std::vector<Fish*> target) {
    Fish* src = original ? original : this;
    J json = hostile->on_damaged(src, target[0], 120).value();
    json["skill"] = skill_to_json("crit", target, hostile->player_id, 120, 1);
    // for(auto enemy : target) {
    //     int id = enemy->pos;
    //     for(auto hits : json["hit"]) {
    //         if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
    //             for(J& targets : json["skill"]["targets"]) {
    //                 if(targets["pos"].asInt64() == id) {
    //                     targets["value"] = hits["value"].asInt64();
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
    return json;
}

// ------------------------8-----------------------
/*
 *被动：行动回合恢复自身 5%血量
 *主动：选择一名队友（可以是自己），令其下次被攻击时减免 70%伤害
*/
opt Octopus::take_damage(Fish *src, int dmg)
{
    J json = Fish::take_damage(src, dmg).value();
    //if(state != FishState::DEAD) {
    if (hp > 0){
        json["passive"].append(passive_to_json("heal", (original?original:this)->pos, ally->player_id, heal_hp(20), update_timestamp()));
    }
    return json;
}

opt Octopus::special(Fish* self, Fish *target)
{
    target->add_buff(FishBuff::SHIELD);
    atk += 20;
    return J();
    // target->buff |= FishBuff::SHIELD;
}

optional<Json::Value> Octopus::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    debug_msg("Octopus skill start");
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    //if(my_list.size() != 1 || enemy_list.size() != 0) return nullopt;
    if (my_list.size() != 1){
        if (my_list.size() == 0) error_type = Fish::friend_target_empty;
        else error_type = Fish::friend_target_too_much;
        return nullopt;
    }
    if (enemy_list.size() != 0){
        error_type = Fish::enemy_target_not_empty;
        return nullopt;
    }
    //if (my_list[0]->state == FishState::DEAD) return nullopt;
    
    J json = Octopus::special(this, my_list[0]).value();
    json["skill"] = skill_to_json("subtle", std::vector<Fish*>(), ally->player_id, 0, 1);
    return json;
}

// ------------------------9-----------------------
/*
 *被动：受伤回20血量
 *主动：对场上血量最少的角色造成 120%的暴击伤害，当目标血量低于其生命值 40% 时改为造成 140%的暴击伤害
 */
opt Whiteshark::take_damage(Fish *src, int dmg)
{
    J json = Fish::take_damage(src, dmg).value();
    //if(state != FishState::DEAD) {
    if (hp > 0){
        json["passive"].append(passive_to_json("heal", (original?original:this)->pos, ally->player_id, heal_hp(20), update_timestamp()));
    }
    return json;
}

opt Whiteshark::special(Fish* self, std::pair<Fish*, int>& effect, Fish* target)
{
    debug_msg("Whiteshark special Start");
    /*Fish* target = nullptr;
    int min_hp = 114514;
    assert(self != nullptr);
    auto e = self->hostile->get_fishs();
    for (auto enemy : e)
    {
        if(enemy->hp <= min_hp && enemy->state != DEAD)
        {
            target = enemy;
            min_hp = enemy->hp;
        }
    }*/
    float addition = 1.2;
    effect.second = self->atk * 1.2;
    if(target->hp < target->max_hp * 0.4) {
        addition = 1.4;
        effect.second = self->atk * 1.4;
    }
    debug_msg("Whiteshark special on_damaged before");
    // J json = self->hostile->on_damaged(self, target, atk * addition).value();
    J json = self->hostile->on_damaged(self, target, effect.second).value();
    debug_msg("Whiteshark special on_damaged after");
    effect.first = target;
    return json;
}

optional<Json::Value> Whiteshark::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    debug_msg("Whiteshark skill Start");
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    //if(my_list.size() != 0 || enemy_list.size() != 1) return nullopt;
    if (my_list.size() != 0){
        error_type = Fish::friend_target_not_empty;
        return nullopt;
    }
    if (enemy_list.size() != 1){
        if (enemy_list.size() == 0) error_type = Fish::enemy_target_empty;
        else error_type = Fish::enemy_target_too_much;
        return nullopt;
    }
    for(auto fish : hostile->get_fishs())
        if (fish->state != FishState::DEAD && enemy_list[0]->hp > fish->hp){
            error_type = Fish::enemy_target_hp_not_lowest;
            return nullopt;
        }
    //if (enemy_list[0]->state == FishState::DEAD) return nullopt;
    
    Fish* src = original ? original : this;
    std::pair<Fish*, int> effect;
    debug_msg("Whiteshark skill special before");

    J json = Whiteshark::special(src, effect, enemy_list[0]).value();

    debug_msg("Whiteshark skill special after");

    std::vector<Fish*> target;
    target.push_back(effect.first);
    json["skill"] = skill_to_json("crit", target, hostile->player_id, effect.second, 1);
    debug_msg("Whiteshark skill to json after");
    // for(auto enemy : target) {
    //     int id = enemy->pos;
    //     for(auto hits : json["hit"]) {
    //         if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
    //             for(J& targets : json["skill"]["targets"]) {
    //                 if(targets["pos"].asInt64() == id) {
    //                     targets["value"] = hits["value"].asInt64();
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
    return json;
}

// ------------------------10-----------------------
/*
 *被动：死亡时会爆炸，造成攻击者最大生命值 10%的伤害；血量低于 50%时，自身伤害提高 30%
 *主动：对场上血量最少的角色造成 120%的暴击伤害，当目标血量低于其生命值 40% 时改为造成 140%的暴击伤害
 */

opt Goblinshark::take_damage(Fish* src, int dmg) {
    J json = Fish::take_damage(src, dmg).value();
    //if(state == FishState::DEAD && src) {
    if (hp <= 0 && src){
        add_json(json, explode(src).value());
    }
    return json;
}

int Goblinshark::get_atk(){
    if(hp * 5 < max_hp) return atk + 15;
    else return atk;
}

opt Goblinshark::special(Fish* self, std::pair<Fish*, int>& effect, Fish* target)
{
    /*Fish* target = nullptr;
    int min_hp = 114514;
    auto e = self->hostile->get_fishs();
    for (auto enemy : e)
    {
        if(enemy->hp <= min_hp && enemy->state != DEAD)
        {
            target = enemy;
            min_hp = enemy->hp;
        }
    }*/
    int hit = 0;
    if(hp * 5 < max_hp) hit = 15;
    float addition = 1.2;
    effect.second = (self->atk + hit) * 1.2;
    if(target->hp < target->max_hp * 0.4) {
        addition = 1.4;
        effect.second = (self->atk + hit) * 1.4;
    }
    // J json = self->hostile->on_damaged(self, target, (atk + hit) * addition).value();
    J json = self->hostile->on_damaged(self, target, effect.second).value();
    effect.first = target;
    return json;
}

optional<Json::Value> Goblinshark::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    /*if(my_list.size() != 0 || enemy_list.size() != 1) return nullopt;
    for(auto fish : hostile->get_fishs())
        if (fish->state != FishState::DEAD && enemy_list[0]->hp > fish->hp) return nullopt;
    if (enemy_list[0]->state == FishState::DEAD) return nullopt;*/
    if (my_list.size() != 0){
        error_type = Fish::friend_target_not_empty;
        return nullopt;
    }
    if (enemy_list.size() != 1){
        if (enemy_list.size() == 0) error_type = Fish::enemy_target_empty;
        else error_type = Fish::enemy_target_too_much;
        return nullopt;
    }
    for(auto fish : hostile->get_fishs())
        if (fish->state != FishState::DEAD && enemy_list[0]->hp > fish->hp){
            error_type = Fish::enemy_target_hp_not_lowest;
            return nullopt;
        }
    
    std::pair<Fish*, int> effect;
    J json = Goblinshark::special(original ? original : this, effect, enemy_list[0]).value();

    std::vector<Fish*> target;
    target.push_back(effect.first);
    json["skill"] = skill_to_json("crit", target, hostile->player_id, effect.second, 1);
    // for(auto enemy : target) {
    //     int id = enemy->pos;
    //     for(auto hits : json["hit"]) {
    //         if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
    //             for(J& targets : json["skill"]["targets"]) {
    //                 if(targets["pos"].asInt64() == id) {
    //                     targets["value"] = hits["value"].asInt64();
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    // }
    return json;
}

optional<Json::Value> Goblinshark::explode(Fish* target) {
    J json;
    json["passive"].append(passive_to_json("explode", (original?original:this)->pos, ally->player_id, 40, update_timestamp()));
    add_json(json, target->take_damage(nullptr, 40).value());
    return json;
}

// ------------------------11-----------------------
// 小丑鱼
/*
 * 被动：行动回合恢复自身 5%血量（见 start_turn()）
 * 主动：为一名队友加 10%的血 (type = 0) 或者给队友群体加 5%的血 (type = 1)
 */
opt Joker::take_damage(Fish *src, int dmg)
{
    J json = Fish::take_damage(src, dmg).value();
    //if(state != FishState::DEAD && hp < 0.3 * max_hp && src != nullptr) {
    if (hp < 0.3 * max_hp && src != nullptr){
        json["passive"].append(passive_to_json("counter", (original?original:this)->pos, ally->player_id, 30, update_timestamp()));
        add_json(json, src->take_damage(nullptr, 30).value());
    }
    return json;
}

opt Joker::special(Fish* tar){
    tar->add_buff(FishBuff::DEFLECT);
    ++special_usage;
    return J();
}

optional<Json::Value> Joker::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    if(!Fish::skill_valid(my_list, enemy_list)) return nullopt;

    /*if(my_list.size() != 1 || special_usage < 3 && enemy_list.size() == 0) return nullopt;
    if(my_list[0]->id == (original?original:this)->id) return nullopt;
    if (my_list[0]->state == FishState::DEAD) return nullopt;*/
    /*if (special_usage < 3){
        for(auto fish : enemy_list)
            if (fish->state == Fish::DEAD) return nullopt;
    }*/
    if (my_list.size() != 1){
        if (my_list.size() == 0) error_type = Fish::friend_target_empty;
        else error_type = Fish::friend_target_too_much;
        return nullopt;
    }
    if (special_usage < 3 && enemy_list.size() == 0){
        error_type = Fish::enemy_target_empty;
        return nullopt;
    }
    if (special_usage < 3 && enemy_list.size() != hostile->count_live_fish()){
        error_type = Fish::enemy_target_not_all;
        return nullopt;
    }
    if (special_usage >= 3 && enemy_list.size() != 0){
        error_type = Fish::enemy_target_not_empty;
        return nullopt;
    }
    if(my_list[0]->id == (original?original:this)->id){
        error_type = Fish::buff_myself;
        return nullopt;
    }
    
    J json = Joker::special(my_list[0]).value();
    json["skill"] = skill_to_json("subtle", std::vector<Fish*>(), ally->player_id, 0, 1);
    if(special_usage <= 3) {
        J bonus = bonus_attack(enemy_list).value();
        add_json(json, bonus);
        json["skill"] = bonus["skill"];
    }
    return json;
}

std::optional<Json::Value> Joker::bonus_attack(std::vector<Fish*> target) {
    Fish* src = original ? original : this;
    J json;
    for(auto f : target) {
        add_json(json, hostile->on_damaged(src, f, 35).value());
    }
    json["skill"] = skill_to_json("aoe", target, hostile->player_id, 35, 1);
    /*for(auto enemy : target) {
        int id = enemy->pos;
        for(auto hits : json["hit"]) {
            if(id == hits["target"].asInt64() && hits["traceable"].asBool()) {
                for(J& targets : json["skill"]["targets"]) {
                    if(targets["pos"].asInt64() == id) {
                        targets["value"] = hits["value"].asInt64();
                        break;
                    }
                }
            }
        }
    }*/
    return json;
}

// ------------------------12-----------------------

// to do
void Imitator::imitate(int id) {
    debug_msg("Imitator imitate id: " + std::to_string(id));
    if (id <= 0 || id >= 12)
        return;
    else
    switch(id){
        case 1:avatar = new Spray(); break;
        case 2:avatar = new Flame(); break;
        case 3:avatar = new Eel(); break;
        case 4:avatar = new Sunfish(); break;
        case 5:avatar = new Blind(); break;
        case 6:avatar = new Flyfish(); break;
        case 7:avatar = new Turtle(); break;
        case 8:avatar = new Octopus(); break;
        case 9:avatar = new Whiteshark(); break;
        case 10:avatar = new Goblinshark(); break;
        case 11:avatar = new Joker(); break;
        default:return;
    }
    avatar->pos = pos;
    avatar->ally = ally;
    avatar->hostile = hostile;
    avatar->original = this;
}

void Imitator::sync() {
    hp = avatar->hp;
    max_hp = avatar->max_hp;
    wound = avatar->wound;
    special_usage = avatar->special_usage;
    shell = avatar->shell;
    atk = avatar->atk;
    def = avatar->def;
    pos = avatar->pos;
    state = avatar->state;
    is_expose = avatar->is_expose;
    buff = avatar->buff; // 被动效果
    error_type = avatar->error_type;
}

opt Imitator::special(Fish* self, std::vector<Fish*> target, int type) {
    return J();
}

optional<Json::Value> Imitator::skill(std::vector<Fish*> my_list, std::vector<Fish*> enemy_list) {
    debug_msg("Imitator skill start");
    debug_msg("Imitator skill avatar id: " + std::to_string(avatar->id));
    opt opt_json = avatar->skill(my_list, enemy_list);
    debug_msg("Imitator skill avatar skill execute after");
    sync();
    if(opt_json == nullopt) {
        return nullopt;
    } else {
        return opt_json.value();
    }
}

int Imitator::heal_hp(int num) {
    int tmp = avatar->heal_hp(num);
    sync();
    return tmp;
}

void Imitator::add_buff(int buff) {
    avatar->add_buff(buff);
    sync();
}
void Imitator::remove_buff(int buff) {
    avatar->remove_buff(buff);
    sync();
}

bool Imitator::check_buff(int _buff) {
    return avatar->check_buff(buff);
}

void Imitator::hp_debuff(int dec) {
    avatar->hp_debuff(dec);
    sync();
}

int Imitator::get_atk(){
    return avatar->get_atk();
}

/*void Imitator::atk_debuff(double rate) {
    avatar->atk_debuff(rate);
    sync();
}*/

opt Imitator::attack(Fish *target) {
    J json = avatar->attack(target).value();
    sync();
    return json;
}

opt Imitator::take_damage(Fish* src, int dmg) {
    debug_msg("Imitator take damage start");
    debug_msg("avatar id: " + std::to_string(avatar->id));
    J json = avatar->take_damage(src, dmg).value();
    debug_msg("Imitator take damage avatar execute after");
    sync();
    return json;
}

std::optional<Json::Value> Imitator::bonus_attack(std::vector<Fish*> target) {
    J json;
    if(avatar->id == Fish::turtle || avatar->id == Fish::joker) {
        if(special_usage <= 3) {
            json = (avatar->bonus_attack(target)).value();
        }
    }
    sync();
    return json;
}

//  TODO: 完成根据id创建对应鱼对象


Fish *create_fish(int id)
{
    if (id <= 0 || id >= 13)
        return nullptr;
    else
    switch(id){
        case 1:return new Spray();
        case 2:return new Flame();
        case 3:return new Eel();
        case 4:return new Sunfish();
        case 5:return new Blind();
        case 6:return new Flyfish();
        case 7:return new Turtle();
        case 8:return new Octopus();
        case 9:return new Whiteshark();
        case 10:return new Goblinshark();
        // case 11:return new Cowfish();
        // case 12:return new Suckerfish();
        //to complete
        // case 13:
        //     return new Pufferfish(100, 25, 0, Fish::kawaii, nullptr, nullptr);
        // case 14:
        //     return new Lion(100, 25, 0, Fish::kawaii, nullptr, nullptr);
        // case 15:
        //     return new Carp(100, 25, 0, Fish::kawaii, nullptr, nullptr);
        // case 16:
        //     return new Doctor(100, 25, 0, Fish::kawaii, nullptr, nullptr);
        case 11:return new Joker();
        case 12:return new Imitator();
        default:return nullptr;
    }
}
