import asyncio
from typing import List
from openai import AsyncClient

import pandas as pd
import numpy as np
import json

import yaml
from argparse import ArgumentParser

from pathlib import Path
import csv

from metrics import cal_metrics, cal_metrics_diff
from customized_model import player_initialization, judge_initialization

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# login_to_your_account_via = "https://education.theb.ai"
# username = "You_USERNAME"
# password = "YOU_PASSWORD"
# login and the website will show all the available models
base_url = "https://education.theb.ai/v1"
api_key = "YOU_API_KEY"
client = AsyncClient(
    base_url=base_url,
    api_key=api_key
)

# ================================= #
# or using official API from OpenAI #
# ================================= #
# from openai import AsyncOpenAI
# # https://github.com/openai/openai-python
# client = AsyncOpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     )


# config file
yaml_file = "./config.yaml"
with open(yaml_file, "r") as f:
    cfg = yaml.safe_load(f)
    print("[Current Config]")
    if cfg["model"]["player_from_API"]:
        print("[Player] Using player model from API| [model name] %s"%(cfg["model"]["player"]))
    else:
        print("[Player] Using player model from others| [model name] %s"%(cfg["model"]["player"]))
    if cfg["model"]["judge_from_API"]:
        print("[Judge] Using judge model from API| [model name] %s"%(cfg["model"]["judge"]))
    else:
        print("[Judge] Using judge model from others| [model name] %s"%(cfg["model"]["judge"]))
# assert False

def log_json(log_dict, i):
    if cfg["log"]["save_log"]:
        if Path(cfg["log"]["log_json"]).exists():
            with open(cfg["log"]["log_json"], "r", encoding="utf-8") as f:
                if i>0:
                    old_log = json.load(f)
                else:
                    old_log = {}
                old_log.update(log_dict)
            with open(cfg["log"]["log_json"], "w", encoding="utf-8") as f:
                json.dump(old_log, f, indent=4, ensure_ascii=False)
        else:
            with open(cfg["log"]["log_json"], "w", encoding="utf-8") as f:
                json.dump(log_dict, f, indent=4, ensure_ascii=False)

    log_dict = {}
    return log_dict


async def generator(model: str, messages: List[dict], stream: bool = True):
    for _ in range(3):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream
            )
            return completion
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(30)


async def game(cfg, player_model_id, judge_model_id, title, story, answer, log_dict, puzzle_id, j_game):

    prompt_player_init = cfg["prompt"]["prompt_player_init"]

    prompt_judge_init = cfg["prompt"]["prompt_judge_init"]

    prompt_puzzle = "The title of puzzle is %s. The story is %s. The answer is %s. Read and fully understand both the provided short story and its answer, ensuring you grasp their logical connections. But do not show answer for the user."%(title, story, answer)

    # initialize the player
    if cfg["model"]["player_from_API"]:
        player_messages = [
            {"role": "assistant", "content": prompt_player_init}
            ]
        player_chat_completion = await generator(model=player_model_id, messages=player_messages, stream=False)
        player_question = player_chat_completion.choices[0].message.content
        print("\033[33mPlayer: %s\033[0m"%(player_question))
        player_messages.append({"role": "assistant", "content": player_question})
        # print(player_question)
        # assert False
    else:
        player, player_conversation = player_initialization(model_id=player_model_id, prompt_role_init=prompt_player_init)
        print("\033[33mPlayer: %s\033[0m"%(player_conversation.messages[-1]["content"]))
        # # "player_conversation.messages" is the same as "player_messages"
        # print(player_conversation.messages)
        # assert False

    # initialize the judge
    if cfg["model"]["judge_from_API"]:
        # tell the judge the rule, puzzle and answer
        judge_messages = [
            {"role": "assistant", "content": prompt_judge_init},
            {"role": "user", "content": prompt_puzzle},
            ]
        judge_chat_completion = await generator(model=judge_model_id, messages=judge_messages, stream=False)
        judge_response = judge_chat_completion.choices[0].message.content
        print("\033[36mJudge: %s\033[0m"%(judge_response))
        judge_messages.append({"role": "assistant", "content": judge_response})

        # the judge shows the puzzle to the player
        judge_messages.append({"role": "user", "content": "I am ready. Please show me the story."})
        print("\033[33mPlayer: %s\033[0m"%("I am ready. Please show me the story."))
        judge_chat_completion = await generator(model=judge_model_id, messages=judge_messages, stream=False)
        judge_response = judge_chat_completion.choices[0].message.content
        print("\033[36mJudge: %s\033[0m"%(judge_response))
        judge_messages.append({"role": "assistant", "content": judge_response})
    else:
        # tell the judge the rule, puzzle and answer
        judge, judge_conversation = judge_initialization(model_id=judge_model_id, prompt_role_init=prompt_judge_init, prompt_puzzle=prompt_puzzle)
        print("\033[36mJudge: %s\033[0m"%(judge_conversation.messages[-1]["content"]))

        # the judge shows the puzzle to the player
        judge_conversation.add_message({"role": "user", "content": "I am ready. Please show me the story."})
        print("\033[33mPlayer: %s\033[0m"%("I am ready. Please show me the story."))
        judge_conversation = judge(judge_conversation, pad_token_id=judge.tokenizer.eos_token_id)
        judge_response = judge_conversation.messages[-1]["content"]
        print("\033[36mJudge: %s\033[0m"%(judge_response))


    log_dict[puzzle_id][j_game]["Puzzle"] = prompt_puzzle
    log_dict[puzzle_id][j_game]["Question"] = judge_response
    

    # game starts
    for rnd in range(cfg["game"]["max_round"]):

        print("\033[35m=================== Round %s ===================\033[0m"%(rnd))
        rnd_id = "Round %s"%(rnd)
        log_dict[puzzle_id][j_game][rnd_id] = {}

        # the player asks question
        if cfg["model"]["player_from_API"]:
            player_messages.append({"role": "user", "content": judge_response})
            player_chat_completion = await generator(model=player_model_id, messages=player_messages, stream=False)
            player_question = player_chat_completion.choices[0].message.content
            player_messages.append({"role": "assistant", "content": player_question})
        else:
            player_conversation.add_message({"role": "user", "content": judge_response})
            player_conversation = player(player_conversation, pad_token_id=player.tokenizer.eos_token_id)
            player_question = player_conversation.messages[-1]["content"]

        print("\033[33mPlayer: %s\033[0m"%(player_question))
        log_dict[puzzle_id][j_game][rnd_id]["Player"] = player_question

        # the judge answers question
        if cfg["model"]["judge_from_API"]:
            judge_messages.append({"role": "user", "content": player_question})
        else:
            judge_conversation.add_message({"role": "user", "content": player_question})

        # judge with multiple times
        judge_list = []
        judge_list_brief = []
        for _ in range(cfg["game"]["judge_times"]):
            
            if cfg["model"]["judge_from_API"]:
                judge_chat_completion = await generator(model=judge_model_id, messages=judge_messages, stream=False)
                judge_response = judge_chat_completion.choices[0].message.content
            else:
                judge_conversation = judge(judge_conversation, pad_token_id=judge.tokenizer.eos_token_id)
                judge_response = judge_conversation.messages[-1]["content"]
                judge_conversation.messages.pop()  # remove the last element (answer) in the message list

            judge_list.append(judge_response.lower())
            if "congratulations" in judge_list[-1]:
                judge_list_brief.append("congratulations")

        print(judge_list)
        print(judge_list_brief)

        # select the most frequent judge answer to alleviate the hallucination
        if len(judge_list_brief) >= cfg["game"]["judge_times"]/2.0:
            judge_response = "congratulations"
        else:
            judge_response = max(judge_list, key=judge_list.count)

        print("\033[36mJudge: %s\033[0m"%(judge_response))
        log_dict[puzzle_id][j_game][rnd_id]["Judge"] = judge_response

        if cfg["model"]["judge_from_API"]:
            judge_messages.pop() # remove the current question
        else:
            # Do not remember the historical question and answer
            judge_conversation.messages.pop() # remove the question

        if "congratulations" in judge_response.lower():
            print("\033[35m=================== Game Over ===================\033[0m")
            hit = 1
            return rnd, hit, log_dict

    hit = 0
    return rnd, hit, log_dict


async def main():
    # prepare data
    # data_path = "./situation_puzzle_merge.xlsx"
    data_path = cfg["path"]["data_path"]

    puzzles = pd.read_excel(data_path, sheet_name=0)
    # puz_set = puzzles.iloc[7:10]
    puz_set = puzzles

    # player_model_id = "llama-3-8b"
    # player_model_id = "llama-3-70b"
    # player_model_id = "dbrx-instruct"
    # player_model_id = "microsoft/WizardLM-2-8x22B"
    # player_model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    player_model_id = cfg["model"]["player"]

    # judge_model_id = "llama-3-70b"
    # judge_model_id = "dbrx-instruct"
    # judge_model_id = "microsoft/WizardLM-2-8x22B"
    # judge_model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    judge_model_id = cfg["model"]["judge"]

    stat_dict = {}  # for calculating the metrics 
    record_list = []  # the result of each sample (id, level, hit, round)
    log_dict = {}  # detailed results for each sample

    for i in range(len(puz_set)):

        # =================== #
        # process puzzle data #
        # =================== #
        print("Situation Puzzle %s"%(i))
        puzzle_id = "puzzle_%s"%(i)
        if puzzle_id not in log_dict.keys():
            log_dict[puzzle_id] = {}

        title = puz_set.iloc[i]["title"]
        story = puz_set.iloc[i]["story"] + " Why?"
        answer = puz_set.iloc[i]["answer"]

        story = story.replace("\n", " ")
        answer = answer.replace("\n", " ")

        ld = puz_set.iloc[i]["level of difficulty"]
        level, difficulty = ld.split(" ")
        level, _ = level.split("/")

        # repeat each puzzle n times
        result_round = 100
        result_hit = 0
        best_id = 0

        for j in range(cfg["game"]["game_times"]):
            # ============= #
            # play the game #
            # ============= #
            log_dict[puzzle_id][j] = {}

            tasks = [game(cfg=cfg, player_model_id=player_model_id, judge_model_id=judge_model_id, title=title, story=story, answer=answer, log_dict=log_dict, puzzle_id=puzzle_id, j_game=j)]

            for g in asyncio.as_completed(tasks):
                result_round_temp, result_hit_temp, log_dict = await g
                result_round_temp += 1 # as it starts from 0, thus we add 1

                # save the current best results
                if result_hit_temp >= result_hit:
                    result_hit = result_hit_temp
                    if result_round_temp <= result_round:
                        result_round = result_round_temp
                        best_id = j

                # write to log.json and clear the dict
                log_json(log_dict, i)

        # calculate the evaluation metrics (1-9 level)
        stat_dict, record_list = cal_metrics(level=level, rnd=result_round, hit=result_hit, stat_dict=stat_dict, record_list=record_list, puz_id=i)
    # calculate the evaluation metrics (difficulty level)
    cal_metrics_diff(stat_dict, record_list)

    with open("log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["puzzle id", "level", "hit", "round"])  # (id, level, hit, round)
        for item in record_list:
            writer.writerow(item)


if __name__ == '__main__':
    asyncio.run(main())