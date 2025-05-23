#pragma once

#include "jsoncpp/json/json.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <random>
#include <array>

enum fish_type
{
	spray = 1,	//射水鱼
	flame,		//喷火鱼
	eel,		//电鳗
	sunfish,	//翻车鱼
	barracuda,	//梭子鱼
	mobula,		//蝠鲼
	turtle,		//海龟
	octopus,	//章鱼
	whiteshark, //大白鲨
	hammerhead, //锤头鲨
	clownfish,	//小丑鱼
	imitator	//拟态鱼
};

enum active_skill
{
	range_attack,	 //射水鱼(spray)+电鳗(eel)
	friend_attack,	 //喷火鱼(flame)+翻车鱼(sunfish)
	critical_attack, //梭子鱼(barracuda)
	reduce_injury,	 //蝠鲼(mobula)+章鱼(octupus)
	limit_critical,	 //海龟(turtle)
	weak_critical,	 //大白鲨(whiteshark)+锤头鲨(hammerhead)
	injury_tansfer	 //小丑鱼(clownfish)
};

enum passive_skill
{
	injury_back,			//射水鱼(spray)+喷火鱼(flame)
	friend_injury_transfer, //电鳗(eel)+翻车鱼(sunfish)
	avoid_injury,			//梭子鱼(barracuda)+蝠鲼(mobula)
	shield,					//海龟(turtle)
	limit_heal,				//章鱼(octopus)+大白鲨(whiteshark)
	dead_boom,				//锤头鲨(hammerhead)
	injury_transfer			//小丑鱼(clownfish)
};

class Fish
{
public:
	int id;
	fish_type type;
	int hp;
	int atk;
	int skill_used = 0;
	active_skill active;
	passive_skill passive;
	Fish() {}
	Fish(int _id, int _hp, int _atk) : id(_id), hp(_hp), atk(_atk) {}
	Fish &operator=(const Fish &_fish)
	{
		id = _fish.id;
		hp = _fish.hp;
		atk = _fish.atk;
		return *this;
	}
};

enum skill_type
{
	aoe,
	infight,
	crit,
	subtle,
	normalattack
};

enum passive_type
{
	counter,
	deflect,
	reduce,
	heal,
	explode
};

const int max_length = 24;

class ActionInfo
{
public:
	//主动
	bool is_skill;										  //是否是技能
	int action_fish;									  //行动的鱼
	int num_friend_targets;								  //友方目标个数
	int num_enemy_targets;								  //敌方目标个数
	std::array<int, 12> friend_targets;					  //友方目标
	std::array<int, 12> enemy_targets;					  //敌方目标
	std::array<int, 12> friend_excepted_injury;			  //友方预期伤害
	std::array<int, 12> enemy_excepted_injury;			  //敌方预期伤害
	std::array<int, 12> friend_expected_injury;			  //友方预期伤害
	std::array<int, 12> enemy_expected_injury;			  //敌方预期伤害
	int num_friend_injury;								  //友方受伤鱼的数量
	int num_enemy_injury;								  //敌方受伤鱼的数量
	std::array<int, max_length> friend_injury_id;		  //友方受伤鱼的位置编号
	std::array<int, max_length> enemy_injury_id;		  //敌方受伤鱼的位置编号
	std::array<int, max_length> friend_injury;			  //友方实际伤害
	std::array<int, max_length> enemy_injury;			  //敌方实际伤害
	std::array<int, max_length> friend_injury_timestamp;  //友方受伤时间戳
	std::array<int, max_length> enemy_injury_timestamp;	  //敌方受伤时间戳
	std::array<bool, max_length> friend_injury_traceable; //友方受伤是否可追踪
	std::array<bool, max_length> enemy_injury_traceable;  //敌方受伤是否可追踪
	skill_type type;
	//被动
	int num_friend_passives;							   //友方被动个数
	int num_enemy_passives;								   //敌方被动个数
	std::array<int, max_length> friend_passives_id;		   //友方触发被动的鱼的id
	std::array<int, max_length> enemy_passives_id;		   //敌方触发被动的鱼的id
	std::array<passive_type, max_length> friend_types;	   //友方触发被动类型
	std::array<passive_type, max_length> enemy_types;	   //敌方触发被动类型
	std::array<double, max_length> friend_passive_value;   //友方减伤比/回血量
	std::array<double, max_length> enemy_passive_value;	   //敌方减伤比/回血量
	std::array<int, max_length> friend_passives_timestamp; //友方触发被动时间戳
	std::array<int, max_length> enemy_passives_timestamp;  //敌方触发被动时间戳

	ActionInfo() : is_skill(false), action_fish(-1), num_friend_targets(0), num_enemy_targets(0),
				   num_friend_injury(0), num_enemy_injury(0), num_friend_passives(0), num_enemy_passives(0) {}
	
	void clear_skill() {
		is_skill = false;
		action_fish = -1;
		num_friend_targets = 0;
		num_enemy_targets = 0;
	}

	void clear_hit() {
		num_friend_injury = 0;
		num_enemy_injury = 0;
	}

	void clear_passive() {
		num_friend_passives = 0;
		num_enemy_passives = 0;
	}

	void clear() {
		clear_skill();
		clear_hit();
		clear_passive();
	}

	bool is_empty() const {
		return is_skill == false && action_fish == -1 && num_friend_targets == 0 &&
			   num_enemy_targets == 0 && num_friend_injury == 0 && num_enemy_injury == 0 &&
			   num_friend_passives == 0 && num_enemy_passives == 0;
	}
};
class AssertInfo
{
public:
	int assertPos;
	int assertContent;
	bool assertResult;

	AssertInfo() : assertPos(0), assertContent(0), assertResult(false) {}

	void clear() {
		assertPos = 0;
		assertContent = 0;
		assertResult = false;
	}

	bool is_empty() const {
		return assertPos == 0 && assertContent == 0 && assertResult == false;
	}
};

//游戏局面
class Game
{
public:
	std::array<const Fish, 4> my_fish;
	std::array<const Fish, 4> enemy_fish;
	const int avatar_id;
	const bool first_mover;
	const ActionInfo enemy_action;
	const ActionInfo my_action;
	const AssertInfo enemy_assert;
	const AssertInfo my_assert;
	const int round;
	const bool round1_win;
	const bool round2_win;
	const bool round3_win;
	const int last_round_finish_reason;
	const bool state_limit_exceed;

	Game(Fish friend1, Fish friend2, Fish friend3, Fish friend4, Fish enemy1, Fish enemy2, Fish enemy3, Fish enemy4, int _avatar_id, bool _first_mover, int round, ActionInfo enemy_action, ActionInfo my_action, AssertInfo enemy_assert, AssertInfo my_assert, bool round1_win, bool round2_win, bool round3_win, int last_round_finish_reason, bool state_limit_exceed) : 
		my_fish{friend1, friend2, friend3, friend4}, enemy_fish{enemy1, enemy2, enemy3, enemy4}, avatar_id(_avatar_id), first_mover(_first_mover), round(round), enemy_action(enemy_action), my_action(my_action), enemy_assert(enemy_assert), my_assert(my_assert), round1_win(round1_win), round2_win(round2_win), round3_win(round3_win), last_round_finish_reason(last_round_finish_reason), state_limit_exceed(state_limit_exceed) {}
};

enum err_type
{
	err_target,
	err_target_out_of_range,
	err_action_type,
	err_action_fish,
	err_action_not_choose,
	err_fish_not_choose,
	success
};

class Action
{
private:
	//-1代表未初始化，-2代表目标为全体
	int actionType = -1;
	int actionFish = -1;
	int enemyTarget = -1;
	int friendTarget = -1;
	Game game;

public:
	Action(Game game) : game(game) {}
	err_type set_action_type(int action_type)
	{
		if (action_type != 0 && action_type != 1)
			return err_action_type;
		this->actionType = action_type;
		return success;
	}
	err_type set_action_fish(int fish_id)
	{
		if (fish_id < 0 || fish_id > 3)
			return err_action_fish;
		this->actionFish = fish_id;
		return success;
	}
	err_type set_enemy_target(int enemy_id)
	{
		if (actionType == -1)
			return err_action_not_choose;
		if (actionFish == -1)
			return err_fish_not_choose;
		enemyTarget = enemy_id;
		if (actionType == 0)
		{
			if (enemy_id < 0 || enemy_id > 3)
				return err_target_out_of_range;
			enemyTarget = enemy_id;
			return success;
		}
		if (actionType == 1)
		{
			err_type error;
			int id = game.my_fish[actionFish].id;
			if (id < 0)
				id = -id;
			if (id == 12)
				id = game.avatar_id;
			switch (id)
			{
			case barracuda:
			case whiteshark:
			case hammerhead:
			case turtle:
				if (enemy_id < 0 || enemy_id > 3)
					error = err_target_out_of_range;
				else
				{
					enemyTarget = enemy_id;
					error = success;
				}
				break;
			case eel:
			case spray:
			case clownfish:
				enemyTarget = -2;
				error = success;
				break;
			case flame:
			case sunfish:
			case mobula:
			case octopus:
				error = err_target;
				break;
			default:
				error = err_action_fish;
				break;
			}
			return error;
		}
		return err_action_type;
	}
	err_type set_friend_target(int friend_id)
	{
		if (actionType == -1)
			return err_action_not_choose;
		if (actionFish == -1)
			return err_fish_not_choose;
		if (actionType == 0)
			return err_target;
		friendTarget = friend_id;
		if (actionType == 1)
		{
			err_type error;
			int id = game.my_fish[actionFish].id;
			if (id < 0)
				id = -id;
			if (id == 12)
				id = game.avatar_id;
			switch (id)
			{
			case barracuda:
			case whiteshark:
			case hammerhead:
			case eel:
			case spray:
				error = err_target;
				break;

			case sunfish:
			case flame:
			case turtle:
			case clownfish:
				if (friend_id == actionFish)
					error = err_target;
				else if (friend_id < 0 || friend_id > 3)
					error = err_target_out_of_range;
				else
				{
					friendTarget = friend_id;
					error = success;
				}
				break;
			case mobula:
			case octopus:
				if (friend_id < 0 || friend_id > 3)
					error = err_target_out_of_range;
				else
				{
					friendTarget = friend_id;
					error = success;
				}
				break;
			default:
				error = err_action_fish;
				break;
			}
			return error;
		}
		return err_action_type;
	}
	int get_action_type()
	{
		return actionType;
	}
	int get_action_fish()
	{
		return actionFish;
	}
	int get_enemy_target()
	{
		return enemyTarget;
	}
	int get_friend_target()
	{
		return friendTarget;
	}
};

class AIClient
{
private:
	std::string recv_msg;
	std::string send_msg;
	Json::Reader reader;
	Json::Value root;
	std::vector<int> remain_fish; //本轮中存活的鱼的编号
	Fish enemy_fish[4];			  //本轮中对手的鱼
	Fish my_fish[4];			  //本轮中选择的鱼
	int avatar_id = -1;			  //本轮中拟态鱼模仿的鱼的编号
	bool first_mover;			  //本轮中是否是先手
	std::ofstream debugger;

	ActionInfo enemy_action;
	ActionInfo my_action;
	AssertInfo enemy_assert;
	AssertInfo my_assert;
	Game *game = nullptr;
	bool round1_win, round2_win, round3_win;
	int current_turn;

	bool state_limit_exceed;
	// init: 0
	// 1: 我方断言结束后结束
	// 2: 我方回合行动结束后结束
	// 3: 敌方断言结束后结束
	// 4: 敌方回合行动结束后结束
	// 6: 不应该出现，如果出现了这个数字请联系@FlagerLee 进行反馈
	int last_round_finish_reason;

public:
	AIClient()
	{
		recv_msg = "";
		send_msg = "";
		//debugger.open("ai_debug_info.txt", std::ios::out);
		//debugger << "======= Debug Info =======" << std::endl;
		//debugger.close();
		current_turn = 0;
		state_limit_exceed = false;
		last_round_finish_reason = 0;
		round1_win = round2_win = round3_win = false;

		Fish init_fish(-1, 0, 0);
		my_fish[0] = my_fish[1] = my_fish[2] = my_fish[3] = init_fish;
		enemy_fish[0] = enemy_fish[1] = enemy_fish[2] = enemy_fish[3] = init_fish;
		ActionInfo init_action;
		enemy_action = my_action = init_action;
		AssertInfo init_assert;
		enemy_assert = my_assert = init_assert;
		updateGame();
	}

	double random(double l, double r)
	{
		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_real_distribution<float> distr(l, r);
		return distr(eng);
	}

	void debug_msg(std::string str)
	{
		debugger.open("ai_debug_info.txt", std::ios::out | std::ios::app);
		debugger << str << std::endl;
		debugger.flush();
		debugger.close();
	}

	void clear_msg()
	{
		recv_msg = "";
		send_msg = "";
		root.clear();
	}

	/*
		发送消息前需要先发送长度
		调用sendLen函数
	*/
	void sendLen(std::string s)
	{
		int len = s.length();
		unsigned char lenb[4];
		lenb[0] = (unsigned char)(len);
		lenb[1] = (unsigned char)(len >> 8);
		lenb[2] = (unsigned char)(len >> 16);
		lenb[3] = (unsigned char)(len >> 24);
		for (int i = 0; i < 4; i++)
			printf("%c", lenb[3 - i]);
	}

	//  发送消息
	void sendrecv_msg(std::string s)
	{
		s.erase(remove(s.begin(), s.end(), '\n'), s.end());
		s.erase(remove(s.begin(), s.end(), '\r'), s.end());
		if ((unsigned char)(s.length()) == '\n' || (unsigned char)(s.length()) == '\r')
			s += " ";
		sendLen(s);
		std::cout << s;
		std::cout.flush();
	}

	//  接收消息
	void listen()
	{
		recv_msg = "";
		while (true)
		{
			recv_msg += getchar();
			if (reader.parse(recv_msg, root))
				break;
		}
		if (!reader.parse(recv_msg, root))
		{ //  不合法判断
			return;
		}
	}

	//sdk函数
	std::vector<int> get_my_remain_fishes()
	{
		//AI可调用，获得自己剩余的鱼的编号
		//srand(time(0));
		//random_shuffle(remain_fish.begin(), remain_fish.end());
		using seed_t = std::default_random_engine::result_type;
		std::default_random_engine eng{static_cast<seed_t>(time(0))};
		std::shuffle(remain_fish.begin(), remain_fish.end(), eng);
		return remain_fish;
	}
	int get_enemy_id(int pos)
	{
		//AI可调用，获得敌方一条鱼的id
		if (pos < 0 || pos >= 4)
			return -2;
		return enemy_fish[pos].id;
	}
	int get_enemy_hp(int pos)
	{
		//AI可调用，获得敌方一条鱼的hp
		if (pos < 0 || pos >= 4)
			return -2;
		return enemy_fish[pos].hp;
	}
	// int get_enemy_atk(int pos)
	// {
	// 	//AI可调用，获得敌方一条鱼的atk
	// 	if (pos < 0 || pos >= 4)
	// 		return -2;
	// 	return enemy_fish[pos].atk;
	// }
	int get_my_id(int pos)
	{
		if (pos < 0 || pos >= 4)
			return -2;
		return my_fish[pos].id;
	}
	int get_my_hp(int pos)
	{
		//AI可调用，获得我方一条鱼的hp
		if (pos < 0 || pos >= 4)
			return -2;
		return my_fish[pos].hp;
	}
	int get_my_atk(int pos)
	{
		//AI可调用，获得我方一条鱼的atk
		if (pos < 0 || pos >= 4)
			return -2;
		return my_fish[pos].atk;
	}

	int get_lowest_health_enemy()
	{
		int enemypos = -1;
		for (int i = 0; i < 4; i++)
		{
			if (get_enemy_hp(i) > 0 && (enemypos == -1 || get_enemy_hp(i) < get_enemy_hp(enemypos)))
				enemypos = i;
		}
		return enemypos;
	}

	std::vector<int> get_lowest_health_enemies()
	{
		int enemypos = -1;
		for (int i = 0; i < 4; i++)
		{
			if (get_enemy_hp(i) > 0 && (enemypos == -1 || get_enemy_hp(i) < get_enemy_hp(enemypos)))
				enemypos = i;
		}
		std::vector<int> enemies;
		enemies.clear();
		for (int i = 0; i < 4; i++)
			if (get_enemy_hp(i) > 0 && get_enemy_hp(i) == get_enemy_hp(enemypos))
				enemies.push_back(i);
		//srand(time(0));
		//random_shuffle(enemies.begin(), enemies.end());
		using seed_t = std::default_random_engine::result_type;
		std::default_random_engine eng{static_cast<seed_t>(time(0))};
		std::shuffle(enemies.begin(), enemies.end(), eng);
		return enemies;
	}

	std::vector<int> get_enemy_living_fishes()
	{
		//AI可调用，获得敌方剩余存活的鱼的编号
		std::vector<int> living_fishes;
		living_fishes.clear();
		for (int i = 0; i < 4; i++)
			if (get_enemy_hp(i) > 0)
				living_fishes.push_back(i);
		//srand(time(0));
		//random_shuffle(living_fishes.begin(), living_fishes.end());
		using seed_t = std::default_random_engine::result_type;
		std::default_random_engine eng{static_cast<seed_t>(time(0))};
		std::shuffle(living_fishes.begin(), living_fishes.end(), eng);
		return living_fishes;
	}

	std::vector<int> get_my_living_fishes()
	{
		//AI可调用，获得自己剩余存活的鱼的编号
		std::vector<int> living_fishes;
		living_fishes.clear();
		for (int i = 0; i < 4; i++)
			if (get_my_hp(i) > 0)
				living_fishes.push_back(i);
		//srand(time(0));
		//random_shuffle(living_fishes.begin(), living_fishes.end());
		using seed_t = std::default_random_engine::result_type;
		std::default_random_engine eng{static_cast<seed_t>(time(0))};
		std::shuffle(living_fishes.begin(), living_fishes.end(), eng);
		return living_fishes;
	}

	std::vector<int> get_my_living_ally(int pos)
	{
		//AI可调用，获得自己一条鱼的其余存活队友
		std::vector<int> living_ally;
		living_ally.clear();
		for (int i = 0; i < 4; i++)
			if (i != pos && get_my_hp(i) > 0)
				living_ally.push_back(i);
		//srand(time(0));
		using seed_t = std::default_random_engine::result_type;
		std::default_random_engine eng{static_cast<seed_t>(time(0))};
		std::shuffle(living_ally.begin(), living_ally.end(), eng);
		return living_ally;
	}

	int get_avatar_id()
	{
		//AI可调用，获得己方拟态鱼本轮模仿的鱼的编号
		return avatar_id;
	}

	bool get_first_mover()
	{
		//AI可调用，获得己方拟态鱼本轮模仿的鱼的编号
		return first_mover;
	}

	int get_skill_used(int pos)
	{
		//AI可调用，获得自己一条鱼的主动技能使用次数
		return my_fish[pos].skill_used;
	}

	Action *auto_valid_action(int pos, Action *action)
	{
		//AI可调用，自动为一个位置的鱼选取合法行动目标，优先高概率使用主动技能，若不可能合法则使用普通攻击
		bool skilled = false;
		int id = get_my_id(pos);
		if (id == 12)
			id = get_avatar_id();
		//srand(time(0));
		//if (rand() % 10 < 3){
		if (random(0, 1) < 0.3)
		{
			action->set_action_type(0);
			action->set_enemy_target((get_enemy_living_fishes())[0]);
			return action;
		}
		switch (id)
		{
		case spray:
		case eel:
			skilled = true;
			action->set_action_type(1);
			action->set_enemy_target(-2);
			break;
		case flame:
		case sunfish:
		{
			auto living_ally = get_my_living_ally(pos);
			if (!living_ally.empty())
			{
				skilled = true;
				action->set_action_type(1);
				action->set_friend_target(living_ally[0]);
			}
			break;
		}
		case barracuda:
		{
			skilled = true;
			action->set_action_type(1);
			action->set_enemy_target((get_enemy_living_fishes())[0]);
			break;
		}
		case mobula:
		case octopus:
			skilled = true;
			action->set_action_type(1);
			action->set_friend_target((get_my_living_fishes())[0]);
			break;
		case turtle:
		{
			auto living_ally = get_my_living_ally(pos);
			if (!living_ally.empty())
			{
				skilled = true;
				action->set_action_type(1);
				action->set_friend_target(living_ally[0]);
				if (get_skill_used(pos) < 3)
					action->set_enemy_target((get_enemy_living_fishes())[0]);
			}
			break;
		}
		case whiteshark:
		case hammerhead:
		{
			skilled = true;
			action->set_action_type(1);
			action->set_enemy_target((get_lowest_health_enemies())[0]);
			break;
		}
		case clownfish:
		{
			auto living_ally = get_my_living_ally(pos);
			if (!living_ally.empty())
			{
				skilled = true;
				action->set_action_type(1);
				action->set_friend_target(living_ally[0]);
				if (get_skill_used(pos) < 3)
					action->set_enemy_target(-2);
			}
			break;
		}
		}
		if (!skilled)
		{
			action->set_action_type(0);
			action->set_enemy_target((get_enemy_living_fishes())[0]);
		}
		return action;
	}

	virtual std::vector<int> Pick(Game game)
	{
		/*
			用户编写AI的函数
			用于完成选鱼上场的操作
		*/
		std::vector<int> v;
		v.clear();
		return v;
	}
	virtual std::pair<int, int> Assert(Game game)
	{
		/*
			用户编写AI的函数
			用于断言操作
			第一个数字返回-1表示不使用断言
		*/
		return std::make_pair(-1, -1);
	}
	virtual Action Act(Game game)
	{ //返回值是行动类型与选取的友方鱼类
		/*
			用户编写AI的函数
			用于行动操作  需要返回一个int类型的二维向量 该向量必须有四行
			第一个向量的第一个数字表示行动类型
			第二个向量的第一个数字表示友方鱼的位置
			若type == 0：
				第三个向量的第一个数字代表需要攻击的敌方鱼位置
			若type == 1：
				第三个向量代表友方鱼目标
				第四个向量代表敌方鱼作用目标

			示例：
			std::vector<std::vector<int>> action;
			action.resize(4);
			然后只需以下标修改各向量内容即可
		*/
		Action action(game);
		return action;
	}

	//解析函数
	void parseGameInfo(Json::Value gameInfo)
	{
		if (gameInfo.isNull() || gameInfo.size() == 0)
			return;
		Json::Value enemyFish = gameInfo["EnemyFish"];
		Json::Value enemyHP = gameInfo["EnemyHP"];
		Json::Value myFish = gameInfo["MyFish"];
		Json::Value myHP = gameInfo["MyHP"];
		Json::Value myATK = gameInfo["MyATK"];

		Fish enemy_fish1(int(enemyFish[0].asInt()), int(enemyHP[0].asInt()), -1);
		Fish enemy_fish2(int(enemyFish[1].asInt()), int(enemyHP[1].asInt()), -1);
		Fish enemy_fish3(int(enemyFish[2].asInt()), int(enemyHP[2].asInt()), -1);
		Fish enemy_fish4(int(enemyFish[3].asInt()), int(enemyHP[3].asInt()), -1);

		Fish my_fish1(int(myFish[0].asInt()), int(myHP[0].asInt()), int(myATK[0].asInt()));
		Fish my_fish2(int(myFish[1].asInt()), int(myHP[1].asInt()), int(myATK[1].asInt()));
		Fish my_fish3(int(myFish[2].asInt()), int(myHP[2].asInt()), int(myATK[2].asInt()));
		Fish my_fish4(int(myFish[3].asInt()), int(myHP[3].asInt()), int(myATK[3].asInt()));

		enemy_fish[0] = enemy_fish1;
		enemy_fish[1] = enemy_fish2;
		enemy_fish[2] = enemy_fish3;
		enemy_fish[3] = enemy_fish4;
		my_fish[0] = my_fish1;
		my_fish[1] = my_fish2;
		my_fish[2] = my_fish3;
		my_fish[3] = my_fish4;
	}
	void parseEnemyActionInfo(Json::Value actionInfo)
	{
		if (actionInfo.isNull() || actionInfo.size() == 0)
		{
			enemy_action.clear_skill();
			enemy_action.clear_hit();
			enemy_action.clear_passive();
			return;
		}
		//action_fish
		if (actionInfo.isMember("ActionFish"))
			enemy_action.action_fish = actionInfo["ActionFish"].asInt();

		//type
		int num_friend_actives = 0, num_enemy_actives = 0;
		if (actionInfo.isMember("skill"))
		{
			Json::Value skill = actionInfo["skill"];
			std::string active_type = skill["type"].asString();
			if (active_type == "aoe")
				enemy_action.type = aoe;
			else if (active_type == "infight")
				enemy_action.type = infight;
			else if (active_type == "crit")
				enemy_action.type = crit;
			else if (active_type == "subtle")
				enemy_action.type = subtle;
			else if (active_type == "normalattack")
				enemy_action.type = normalattack;

			enemy_action.is_skill = skill["isSkill"].asInt();

			//friend_targets && enemy_targets && friend_expected_injury && enemy_expected_injury
			int size = skill["targets"].size();
			if (enemy_action.type != subtle)
				for (int i = 0; i < size; i++)
				{
					Json::Value target = skill["targets"][i];
					if (!target["isEnemy"].asBool())
					{
						enemy_action.enemy_targets[num_enemy_actives] = target["pos"].asInt();
						enemy_action.enemy_excepted_injury[num_enemy_actives] = target["value"].asInt();
						enemy_action.enemy_expected_injury[num_enemy_actives] = target["value"].asInt();
						num_enemy_actives++;
					}
					else
					{
						enemy_action.friend_targets[num_friend_actives] = target["pos"].asInt();
						enemy_action.friend_excepted_injury[num_enemy_actives] = target["value"].asInt();
						enemy_action.friend_expected_injury[num_enemy_actives] = target["value"].asInt();
						num_friend_actives++;
					}
				}
		}
		else enemy_action.clear_skill();
		enemy_action.num_friend_targets = num_friend_actives;
		enemy_action.num_enemy_targets = num_enemy_actives;

		//friend_injury, enemy_injury, friend_injury_id, enemy_injury_id
		int num_friend_hit = 0, num_enemy_hit = 0;
		if (actionInfo.isMember("hit"))
		{
			Json::Value hit = actionInfo["hit"];
			int size = hit.size();
			for (int i = 0; i < size; i++)
			{
				Json::Value target = hit[i];
				if (!target["isEnemy"].asBool())
				{
					enemy_action.enemy_injury_id[num_enemy_hit] = target["target"].asInt();
					enemy_action.enemy_injury[num_enemy_hit] = target["value"].asInt();
					if (target.isMember("time"))
						enemy_action.enemy_injury_timestamp[num_enemy_hit] = target["time"].asInt();
					if (target.isMember("traceable"))
						enemy_action.enemy_injury_traceable[num_enemy_hit] = target["traceable"].asBool();
					num_enemy_hit++;
				}
				else
				{
					enemy_action.friend_injury_id[num_friend_hit] = target["target"].asInt();
					enemy_action.friend_injury[num_friend_hit] = target["value"].asInt();
					if (target.isMember("time"))
						enemy_action.friend_injury_timestamp[num_friend_hit] = target["time"].asInt();
					if (target.isMember("traceable"))
						enemy_action.friend_injury_traceable[num_friend_hit] = target["traceable"].asBool();
					num_friend_hit++;
				}
			}
		}
		else enemy_action.clear_hit();
		enemy_action.num_friend_injury = num_friend_hit;
		enemy_action.num_enemy_injury = num_enemy_hit;

		//friend_passives_id, enemy_passives_id, friend_passives, enemy_passives, friend_types, enemy_types, friend_passive_value, enemy_passive_value
		int num_friend_passives = 0, num_enemy_passives = 0;
		if (actionInfo.isMember("passive"))
		{
			Json::Value passives = actionInfo["passive"];
			int size = passives.size();
			for (int i = 0; i < size; i++)
			{
				Json::Value passive = passives[i];
				if (!passive["isEnemy"].asBool())
				{
					enemy_action.enemy_passives_id[num_enemy_passives] = passive["source"].asInt();
					if (passive.isMember("time"))
						enemy_action.enemy_passives_timestamp[num_enemy_passives] = passive["time"].asInt();
					std::string passiveType = passive["type"].asString();
					if (passiveType == "counter")
						enemy_action.enemy_types[num_enemy_passives] = counter;
					else if (passiveType == "deflect")
						enemy_action.enemy_types[num_enemy_passives] = deflect;
					else if (passiveType == "reduce")
						enemy_action.enemy_types[num_enemy_passives] = reduce;
					else if (passiveType == "heal")
						enemy_action.enemy_types[num_enemy_passives] = heal;
					else if (passiveType == "explode")
						enemy_action.enemy_types[num_enemy_passives] = explode;
					if (passiveType == "reduce" && passive.isMember("value"))
						enemy_action.enemy_passive_value[num_enemy_passives] = passive["value"].asDouble();
					else if (passiveType == "heal" && passive.isMember("value"))
						enemy_action.enemy_passive_value[num_enemy_passives] = passive["value"].asInt();
					num_enemy_passives++;
				}
				else
				{
					enemy_action.friend_passives_id[num_friend_passives] = passive["source"].asInt();
					if (passive.isMember("time"))
						enemy_action.friend_passives_timestamp[num_friend_passives] = passive["time"].asInt();
					std::string passiveType = passive["type"].asString();
					if (passiveType == "counter")
						enemy_action.friend_types[num_friend_passives] = counter;
					else if (passiveType == "deflect")
						enemy_action.friend_types[num_friend_passives] = deflect;
					else if (passiveType == "reduce")
						enemy_action.friend_types[num_friend_passives] = reduce;
					else if (passiveType == "heal")
						enemy_action.friend_types[num_friend_passives] = heal;
					else if (passiveType == "explode")
						enemy_action.friend_types[num_friend_passives] = explode;
					if (passiveType == "reduce" && passive.isMember("value"))
						enemy_action.friend_passive_value[num_friend_passives] = passive["value"].asDouble();
					else if (passiveType == "heal" && passive.isMember("value"))
						enemy_action.friend_passive_value[num_friend_passives] = passive["value"].asInt();
					num_friend_passives++;
				}
			}
		}
		else enemy_action.clear_passive();
		enemy_action.num_friend_passives = num_friend_passives;
		enemy_action.num_enemy_passives = num_enemy_passives;
	}
	void parseMyActionInfo(Json::Value actionInfo)
	{
		if (actionInfo.isNull() || actionInfo.size() == 0)
		{
			my_action.clear_skill();
			my_action.clear_hit();
			my_action.clear_passive();
			return;
		}
		//action_fish
		if (actionInfo.isMember("ActionFish"))
			my_action.action_fish = actionInfo["ActionFish"].asInt();

		//type
		int num_friend_actives = 0, num_enemy_actives = 0;
		if (actionInfo.isMember("skill"))
		{
			Json::Value skill = actionInfo["skill"];
			std::string active_type = skill["type"].asString();
			if (active_type == "aoe")
				my_action.type = aoe;
			else if (active_type == "infight")
				my_action.type = infight;
			else if (active_type == "crit")
				my_action.type = crit;
			else if (active_type == "subtle")
				my_action.type = subtle;
			else if (active_type == "normalattack")
				my_action.type = normalattack;

			my_action.is_skill = skill["isSkill"].asInt();

			//friend_targets && enemy_targets && friend_expected_injury && enemy_expected_injury
			int size = skill["targets"].size();
			if (my_action.type != subtle)
				for (int i = 0; i < size; i++)
				{
					Json::Value target = skill["targets"][i];
					if (target["isEnemy"].asBool())
					{
						my_action.enemy_targets[num_enemy_actives] = target["pos"].asInt();
						my_action.enemy_excepted_injury[num_enemy_actives] = target["value"].asInt();
						my_action.enemy_expected_injury[num_enemy_actives] = target["value"].asInt();
						num_enemy_actives++;
					}
					else
					{
						my_action.friend_targets[num_friend_actives] = target["pos"].asInt();
						my_action.friend_excepted_injury[num_enemy_actives] = target["value"].asInt();
						my_action.friend_expected_injury[num_enemy_actives] = target["value"].asInt();
						num_friend_actives++;
					}
				}
		}
		else my_action.clear_skill();
		my_action.num_friend_targets = num_friend_actives;
		my_action.num_enemy_targets = num_enemy_actives;

		//friend_injury, enemy_injury, friend_injury_id, enemy_injury_id
		int num_friend_hit = 0, num_enemy_hit = 0;
		if (actionInfo.isMember("hit"))
		{
			Json::Value hit = actionInfo["hit"];
			int size = hit.size();
			for (int i = 0; i < size; i++)
			{
				Json::Value target = hit[i];
				if (target["isEnemy"].asBool())
				{
					my_action.enemy_injury_id[num_enemy_hit] = target["target"].asInt();
					my_action.enemy_injury[num_enemy_hit] = target["value"].asInt();
					if (target.isMember("time"))
						my_action.enemy_injury_timestamp[num_enemy_hit] = target["time"].asInt();
					if (target.isMember("traceable"))
						my_action.enemy_injury_traceable[num_enemy_hit] = target["traceable"].asBool();
					num_enemy_hit++;
				}
				else
				{
					my_action.friend_injury_id[num_friend_hit] = target["target"].asInt();
					my_action.friend_injury[num_friend_hit] = target["value"].asInt();
					if (target.isMember("time"))
						my_action.friend_injury_timestamp[num_friend_hit] = target["time"].asInt();
					if (target.isMember("traceable"))
						my_action.friend_injury_traceable[num_friend_hit] = target["traceable"].asBool();
					num_friend_hit++;
				}
			}
		}
		else my_action.clear_hit();
		my_action.num_friend_injury = num_friend_hit;
		my_action.num_enemy_injury = num_enemy_hit;

		//friend_passives_id, enemy_passives_id, friend_passives, enemy_passives, friend_types, enemy_types, friend_passive_value, enemy_passive_value
		int num_friend_passives = 0, num_enemy_passives = 0;
		if (actionInfo.isMember("passive"))
		{
			Json::Value passives = actionInfo["passive"];
			int size = passives.size();
			for (int i = 0; i < size; i++)
			{
				Json::Value passive = passives[i];
				if (passive["isEnemy"].asBool())
				{
					my_action.enemy_passives_id[num_enemy_passives] = passive["source"].asInt();
					if (passive.isMember("time"))
						my_action.enemy_passives_timestamp[num_enemy_passives] = passive["time"].asInt();
					std::string passiveType = passive["type"].asString();
					if (passiveType == "counter")
						my_action.enemy_types[num_enemy_passives] = counter;
					else if (passiveType == "deflect")
						my_action.enemy_types[num_enemy_passives] = deflect;
					else if (passiveType == "reduce")
						my_action.enemy_types[num_enemy_passives] = reduce;
					else if (passiveType == "heal")
						my_action.enemy_types[num_enemy_passives] = heal;
					else if (passiveType == "explode")
						my_action.enemy_types[num_enemy_passives] = explode;
					if (passiveType == "reduce" && passive.isMember("value"))
						my_action.enemy_passive_value[num_enemy_passives] = passive["value"].asDouble();
					else if (passiveType == "heal" && passive.isMember("value"))
						my_action.enemy_passive_value[num_enemy_passives] = passive["value"].asInt();
					num_enemy_passives++;
				}
				else
				{
					my_action.friend_passives_id[num_friend_passives] = passive["source"].asInt();
					if (passive.isMember("time"))
						my_action.friend_passives_timestamp[num_friend_passives] = passive["time"].asInt();
					std::string passiveType = passive["type"].asString();
					if (passiveType == "counter")
						my_action.friend_types[num_friend_passives] = counter;
					else if (passiveType == "deflect")
						my_action.friend_types[num_friend_passives] = deflect;
					else if (passiveType == "reduce")
						my_action.friend_types[num_friend_passives] = reduce;
					else if (passiveType == "heal")
						my_action.friend_types[num_friend_passives] = heal;
					else if (passiveType == "explode")
						my_action.friend_types[num_friend_passives] = explode;
					if (passiveType == "reduce" && passive.isMember("value"))
						my_action.friend_passive_value[num_friend_passives] = passive["value"].asDouble();
					else if (passiveType == "heal" && passive.isMember("value"))
						my_action.friend_passive_value[num_friend_passives] = passive["value"].asInt();
					num_friend_passives++;
				}
			}
		}
		else my_action.clear_passive();
		my_action.num_friend_passives = num_friend_passives;
		my_action.num_enemy_passives = num_enemy_passives;
	}
	void parseEnemyAssert(Json::Value assertInfo)
	{
		if (assertInfo.isNull() || assertInfo.size() == 0)
		{
			enemy_assert.assertPos = 0;
			enemy_assert.assertContent = 0;
			enemy_assert.assertResult = false;
			return;
		}
		enemy_assert.assertPos = assertInfo["AssertPos"].asInt();
		enemy_assert.assertContent = assertInfo["AssertContent"].asInt();
		enemy_assert.assertResult = assertInfo["AssertResult"].asBool();
	}
	void parseMyAssert(Json::Value assertInfo)
	{
		if (assertInfo.isNull() || assertInfo.size() == 0)
		{
			my_assert.assertPos = 0;
			my_assert.assertContent = 0;
			my_assert.assertResult = false;
			return;
		}
		my_assert.assertPos = assertInfo["AssertPos"].asInt();
		my_assert.assertContent = assertInfo["AssertContent"].asInt();
		my_assert.assertResult = assertInfo["AssertResult"].asBool();
	}

	//更新函数
	void updateGame()
	{
		if (game != nullptr)
			delete (game);
		game = new Game(my_fish[0], my_fish[1], my_fish[2], my_fish[3], enemy_fish[0], enemy_fish[1], enemy_fish[2], enemy_fish[3], avatar_id, first_mover, current_turn, enemy_action, my_action, enemy_assert, my_assert, round1_win, round2_win, round3_win, last_round_finish_reason, state_limit_exceed);
	}

	//动作函数
	void Action_Pick()
	{
		current_turn++;

		remain_fish.clear();
		for (auto x : root["RemainFishs"])
			remain_fish.push_back(x.asInt());
		first_mover = root["FirstMover"].asInt();
		updateGame();
		std::vector<int> pickfish = Pick(*game);
		for (int i = 0; i < 4; i++)
			my_fish[i].skill_used = 0;
		avatar_id = -1;
		Json::Value operation;
		operation.clear();
		operation["Action"] = "Pick";
		operation["ChooseFishs"].resize(0);
		for (auto x : pickfish)
		{
			if (x > 12)
			{
				operation["ImitateFish"] = x - 12;
				avatar_id = x - 12;
				operation["ChooseFishs"].append(12);
			}
			else
				operation["ChooseFishs"].append(x);
		}
		updateGame();
		Json::FastWriter writer;
		send_msg = writer.write(operation);
	}
	void Action_Assert()
	{
		parseGameInfo(root["GameInfo"]);
		parseEnemyActionInfo(root["EnemyAction"]);
		parseMyActionInfo(root["MyAction"]);
		parseEnemyAssert(root["EnemyAssert"]);
		updateGame();
		std::pair<int, int> pi = Assert(*game);
		int assert_id = pi.first;
		Json::Value operation;
		operation.clear();
		if (assert_id < 0 || assert_id >= 4 || enemy_fish[assert_id].id != -1)
			operation["Action"] = "Null";
		else
		{
			operation["Action"] = "Assert";
			operation["Pos"] = assert_id;
			operation["ID"] = pi.second;
		}
		Json::FastWriter writer;
		send_msg = writer.write(operation);
	}
	void Action_Action()
	{
		parseGameInfo(root["GameInfo"]);
		parseMyAssert(root["AssertReply"]);
		updateGame();
		Json::Value operation;
		operation.clear();
		Action my_act = Act(*game);
		operation["Action"] = "Action";
		operation["Type"] = my_act.get_action_type();
		operation["MyPos"] = my_act.get_action_fish();
		if (my_act.get_action_type() == 0)
			operation["EnemyPos"] = my_act.get_enemy_target();
		else if (my_act.get_action_type() == 1)
		{
			my_fish[my_act.get_action_fish()].skill_used++;
			if (my_act.get_enemy_target() == -2)
			{
				//aoe
				operation["EnemyList"].resize(0);
				for (int i = 0; i < 4; i++)
					if (get_enemy_hp(i) > 0)
						operation["EnemyList"].append(i);
			}
			else if (my_act.get_enemy_target() != -1)
			{
				operation["EnemyList"].resize(0);
				operation["EnemyList"].append(my_act.get_enemy_target());
			}
			else
				operation["EnemyList"].resize(0);
			if (my_act.get_friend_target() != -1)
			{
				operation["MyList"].resize(0);
				operation["MyList"].append(my_act.get_friend_target());
			}
			else
				operation["MyList"].resize(0);
		}
		Json::FastWriter writer;
		send_msg = writer.write(operation);
	}
	void Action_Finish()
	{
		if (current_turn == 1)
			round1_win = root["Result"].asString() == "Win" ? true : false;
		else if (current_turn == 2)
			round2_win = root["Result"].asString() == "Win" ? true : false;
		else if (current_turn == 3)
			round3_win = root["Result"].asString() == "Win" ? true : false;

		if (root.isMember("MyAssert"))
			parseMyAssert(root["MyAssert"]);
		else my_assert.clear();
		if (root.isMember("EnemyAction"))
			parseEnemyActionInfo(root["EnemyAction"]);
		else enemy_action.clear();
		if (root.isMember("MyAction"))
			parseMyActionInfo(root["MyAction"]);
		else my_action.clear();
		if (root.isMember("EnemyAssert"))
			parseEnemyAssert(root["EnemyAssert"]);
		else enemy_assert.clear();
		if (!root.isMember("isStateLimitExceed")) state_limit_exceed = false;
		else if (root["isStateLimitExceed"].asBool()) state_limit_exceed = true;
		else state_limit_exceed = false;
		if(root.isMember("MyAssert") && !root["MyAssert"].isNull())
		{
			// 我方断言成功并战胜对面
			last_round_finish_reason = 1;
		}
		else 
		{
			if(root.isMember("MyAction") && !root["MyAction"].isNull())
			{
				if(root.isMember("EnemyAssert") && !root["EnemyAssert"].isNull())
				{
					if(root.isMember("EnemyAction") && !root["EnemyAction"].isNull())
						last_round_finish_reason = 4; //敌方行动阶段赢得比赛
					else
						last_round_finish_reason = 3; //敌方断言阶段赢得比赛
				}
				else
					last_round_finish_reason = 2; //我方行动阶段赢得比赛
			}
			else
				last_round_finish_reason = 6;
		}
		updateGame();
		send_msg = "{\"Action\":\"Finish\"}";
	}

	//回合制循环
	void run()
	{
		while (true)
		{
			clear_msg();
			listen();
			if (root["Action"].asString() == "Pick")
				Action_Pick();
			else if (root["Action"].asString() == "Assert")
				Action_Assert();
			else if (root["Action"].asString() == "Action")
				Action_Action();
			else if (root["Action"].asString() == "Finish")
				Action_Finish();
			sendrecv_msg(send_msg);
		}
	}
};
