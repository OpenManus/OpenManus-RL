#include "fishset.h"

std::ofstream debuggerfs;
void debugmsg(std::string str){
    return;
    debuggerfs.open("debug_info.txt", std::ios::out | std::ios::app);
    debuggerfs << str << std::endl;
    debuggerfs.close();
}

// to do
FishSet::FishSet(int flag){
    fishs.clear();
    if (flag){
        // for(int i=1; i<=6; i++) fishs.push_back(create_fish(Fish::Big));
        // for(int i=1; i<=12; i++) fishs.push_back(create_fish(Fish::Small));
        for(int i=1; i<=12; i++) fishs.push_back(create_fish(i));
    }
}

FishSet::FishSet(const FishSet& base){
    fishs.clear();
    for(auto fish : base.get_fishs()){
        assert(fish != nullptr);
        Fish* _fish = create_fish(fish->id);
        fishs.push_back(_fish);
    }
}

/*
    按顺序将集合中的鱼输出为 JSON 数组
    样例：
        [
            fish1,
            fish2,
            ...
        ]
*/
Json::Value FishSet::to_json() const{
    Json::Value json;
    json.resize(0);
    for(auto fish : fishs)
        json.append(fish->to_json());
    return json;
}

std::vector<Fish*> FishSet::get_fishs() const{
    return fishs;
}

int FishSet::get_size(){
    return fishs.size();
}

bool FishSet::empty(){
    return fishs.empty();
}

void FishSet::clear(){
    fishs.clear();
}

void FishSet::add(Fish* fish){
    if (fish != nullptr){
        fishs.push_back(fish);
    }
}

void FishSet::add(int id){
    if (Fish::is_valid_fish_id(id)){
        fishs.push_back(create_fish(id));
    }
}

void FishSet::remove(int id){
    if (Fish::is_valid_fish_id(id)){
        for(auto it = fishs.begin(); it != fishs.end(); it++){
            if ((*it)->id == id){
                fishs.erase(it);
                return;
            }
        }
    }
}

void FishSet::to_fight(){
    /*for(auto fish : fishs)
        fish->state = Fish::FIGHT;*/
    for(int i = 0;i < 4;i ++){
        auto fish = fishs[i];
        fish->state = Fish::FIGHT;
        fish->pos = i;
    }
}

void FishSet::to_dead(){
    for(auto fish : fishs)
        fish->state = Fish::DEAD;
    clear();
}

void FishSet::update_state(){
    for(auto fish : fishs)
        if (fish->hp <= 0)
            fish->state = Fish::DEAD;
}

void FishSet::hp_debuff(int dec){
    for(auto fish : fishs)
        fish->hp_debuff(dec);
}

int FishSet::count_live_fish(){
    int cnt = 0;
    for(auto fish : fishs)
        if (fish->state != Fish::DEAD) cnt++;
    return cnt;
}

int FishSet::living_fish_count() const {
    int ret = 0;
    for(auto f: fishs){
        if(f->hp > 0) ++ret;
    }
    return ret;
}

int FishSet::hp_sum() const {
    int ret = 0;
    for(auto f: fishs){
        if(f->hp > 0) ret += f->hp;
    }
    return ret;
}

int FishSet::hp_max() const {
    int ret = 0;
    for(auto f: fishs)
        ret = std::max(ret, f->hp);
    return ret;
}

bool FishSet::is_all_dead(){
    for(auto fish : fishs)
        if (fish->state != Fish::DEAD) return false;
    return true;
}

bool FishSet::count(Fish* tar) {
    for(auto i:fishs)
        if(i == tar) return 1;
    return 0;
}

int FishSet::update_timestamp(){
    timestamp ++;
    another->timestamp ++;
    return timestamp;
}

void FishSet::set_fishset(FishSet* _ally, FishSet* _hostile) {
    for(auto i: fishs) {
        i->set_fishset(_ally, _hostile);
    }
}

/*
在此处处理的技能：
狮子鱼(Lion), 喷火鱼(flame): 群体反伤
*/
std::optional<Json::Value> FishSet::on_damaged(Fish* src, Fish* target, int dmg) {
    // debugmsg("FishSet on damaged start");
    // Json::Value json = target->take_damage(src, dmg).value();
    assert(target != nullptr);
    // debugmsg(std::to_string(target->id));
    auto tg = target->take_damage(src, dmg);
    // debugmsg("FishSet on damaged target take damage before");
    assert(tg);
    Json::Value json = tg.value();
    // debugmsg("FishSet on damaged tg value before");
    
    if(src && target->hp < target->max_hp*0.3){
        for(auto fish : fishs) {
            int nid;
            if (fish->id == 12) nid = (dynamic_cast<Imitator*>(fish))->avatar->id;
            else nid = fish->id;
            if((nid == Fish::spray || nid == Fish::flame) && !(fish->state == Fish::DEAD) && target->pos != fish->pos) {
                json["passive"].append(passive_to_json("counter", target->pos, target->ally->player_id, 30, update_timestamp()));
                add_json(json, src->take_damage(nullptr, 30).value());
            }
        }
    }
    return json;
}

void FishSet::start_turn() {
    for(auto f: fishs) {
        f -> start_turn();
    }
}
