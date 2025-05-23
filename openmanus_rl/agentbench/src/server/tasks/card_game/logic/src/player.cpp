#include "player.h"

Player::Player(int _id):id(_id){
    my_fish = FishSet(1);
    my_fish.player_id = _id;
    fight_fish = FishSet();
    fight_fish.player_id = _id;
}

Player::Player(int _id, FishSet _my_fish):id(_id),my_fish(_my_fish){
    my_fish = FishSet(1);
    fight_fish = FishSet();
}

Player::Player(const Player &p){
    id = p.id;
    type = p.type;
    my_fish = p.my_fish;
    fight_fish = p.fight_fish;
}

/*
    按照成员变量定义顺序转成 JSON
    样例：
        {
            "id": id,
            "type": type,
            "my_fish": my_fish,
            "fight_fish": fight_fish
        }
*/
Json::Value Player::to_json() const{
    Json::Value json;
    json["id"] = id;
    json["type"] = type;
    json["my_fish"] = my_fish.to_json();
    json["fight_fish"] = fight_fish.to_json();
    return json;
}

int Player::get_id(){
    return id;
}

std::vector<Fish*> Player::get_fishs() const{
    return my_fish.get_fishs();
}

std::vector<Fish*> Player::get_fight_fishs() const{
    return fight_fish.get_fishs();
}

int Player::get_size(){
    return my_fish.get_size();
}

bool Player::empty(){
    return my_fish.empty();
}

void Player::clear(){
    my_fish.clear();
}

void Player::add(Fish* fish){
    my_fish.add(fish);
}

void Player::add(int id){
    my_fish.add(id);
}

void Player::remove(int id){
    my_fish.remove(id);
}

void Player::to_war(int id){
    my_fish.remove(id);
    fight_fish.add(id);
}