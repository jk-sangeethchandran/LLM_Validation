#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_model_validation_runner.py

Unified FSP validation runner for multiple instructor and target models.

What it does:
1. Lets you choose one or more instructor models and one or more target models before any data is sent.
2. Uses the official IPIP-NEO-300 key embedded from your "Conversational And Non-verbal.py" script.
3. The instructor model converts each participant's full 300-item questionnaire profile into a detailed,
   facet-by-facet personality control prompt.
4. The target model receives:
      - the instructor-generated personality prompt
      - calibration statistics from the first 120 items
      - labelled first-120 item examples
   Then it answers items 121-300.
5. Runs all selected instructor × target combinations.
6. Caches instructor prompts, system prompts, target batches, and OpenAI prompt-cache hints where supported.
7. Exports Excel/CSV/JSONL results and publication-friendly plots.

Install in the same venv:
    python -m pip install -U pandas numpy scipy scikit-learn matplotlib openpyxl python-dotenv groq openai "openai[realtime]" google-genai

Expected files beside this script, unless overridden:
    .env
    IPIP_NEO_300.csv   OR   prosocial_antisocial_ipip300_answers.csv

Relevant .env variables:
    OPENAI_API_KEY=...
    GROQ_API_KEY=...                 # only needed if using llama-3.1-8b-instant
    GROQ_API_KEY_TUD=...             # optional alternative
    GEMINI_API_KEY=...               # Google AI Studio / Gemini API key, only needed for Gemma
    GOOGLE_AI_STUDIO_API_KEY=...     # optional alternative env name
    GOOGLE_API_KEY=...               # optional alternative env name

Gemma is called through Google AI Studio / Gemini API using the google-genai SDK,
not through a local workstation endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import mean_squared_error

try:
    from openpyxl import load_workbook
    from openpyxl.drawing.image import Image as XLImage
except Exception:
    load_workbook = None
    XLImage = None


# =============================================================================
# Paths and runtime constants
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

PROMPT_VERSION = "multi_model_fsp_validation_v2_google_ai_studio_gemma"
SELECTION_FILE = SCRIPT_DIR / ".validation_model_selection.json"
CACHE_DIR = SCRIPT_DIR / ".validation_cache"
OUTPUT_ROOT = SCRIPT_DIR / "validation_outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)

DEFAULT_MAX_PARTICIPANTS = 10
DEFAULT_BATCH_SIZE = 30
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none").strip() or "none"
DEFAULT_OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1800"))
DEFAULT_INSTRUCTOR_MAX_TOKENS = int(os.getenv("INSTRUCTOR_MAX_OUTPUT_TOKENS", "1800"))
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1800"))
DEFAULT_GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1800"))

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY_TUD") or os.getenv("GROQ_API_KEY") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

TRAIN_ITEMS = list(range(1, 121))
TEST_ITEMS = list(range(121, 301))
ALL_ITEMS = list(range(1, 301))
TRAIN_COLS = [f"i{n}" for n in TRAIN_ITEMS]
TEST_COLS = [f"i{n}" for n in TEST_ITEMS]
ALL_COLS = [f"i{n}" for n in ALL_ITEMS]

DOMAIN_ORDER = ["O", "C", "E", "A", "N"]
DOMAIN_FULL_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

LABEL_MAP_FORMAL = {
    1: "Very Inaccurate",
    2: "Moderately Inaccurate",
    3: "Neither Accurate Nor Inaccurate",
    4: "Moderately Accurate",
    5: "Very Accurate",
    0: "Neither Accurate Nor Inaccurate",
}

SCORE_LABEL_FOR_INSTRUCTOR = {
    1: "almost never / strongly inaccurate",
    2: "rarely / moderately inaccurate",
    3: "sometimes / neither accurate nor inaccurate",
    4: "often / moderately accurate",
    5: "almost always / very accurate",
}

# Embedded from your Conversational And Non-verbal.py official-key script.
# Tuple format: (item_number, domain, facet, reverse_scored, item_text)
ITEM_KEY_300 = [(1, 'N', 'Anxiety', False, 'Worry about things.'),
 (2, 'E', 'Friendliness', False, 'Make friends easily.'),
 (3, 'O', 'Imagination', False, 'Have a vivid imagination.'),
 (4, 'A', 'Trust', False, 'Trust others.'),
 (5, 'C', 'Self-Efficacy', False, 'Complete tasks successfully.'),
 (6, 'N', 'Anger', False, 'Get angry easily.'),
 (7, 'E', 'Gregariousness', False, 'Love large parties.'),
 (8, 'O', 'Artistic Interests', False, 'Believe in the importance of art.'),
 (9, 'A', 'Morality', False, 'Would never cheat on my taxes.'),
 (10, 'C', 'Orderliness', False, 'Like order.'),
 (11, 'N', 'Depression', False, 'Often feel blue.'),
 (12, 'E', 'Assertiveness', False, 'Take charge.'),
 (13, 'O', 'Emotionality', False, 'Experience my emotions intensely.'),
 (14, 'A', 'Altruism', False, 'Make people feel welcome.'),
 (15, 'C', 'Dutifulness', False, 'Try to follow the rules.'),
 (16, 'N', 'Self-Consciousness', False, 'Am easily intimidated.'),
 (17, 'E', 'Activity Level', False, 'Am always busy.'),
 (18, 'O', 'Adventurousness', False, 'Prefer variety to routine.'),
 (19, 'A', 'Cooperation', False, 'Am easy to satisfy.'),
 (20, 'C', 'Achievement-Striving', False, 'Go straight for the goal.'),
 (21, 'N', 'Immoderation', False, 'Often eat too much.'),
 (22, 'E', 'Excitement-Seeking', False, 'Love excitement.'),
 (23, 'O', 'Intellect', False, 'Like to solve complex problems.'),
 (24, 'A', 'Modesty', False, 'Dislike being the center of attention.'),
 (25, 'C', 'Self-Discipline', False, 'Get chores done right away.'),
 (26, 'N', 'Vulnerability', False, 'Panic easily.'),
 (27, 'E', 'Cheerfulness', False, 'Radiate joy.'),
 (28, 'O', 'Liberalism', False, 'Tend to vote for liberal political candidates.'),
 (29, 'A', 'Sympathy', False, 'Sympathize with the homeless.'),
 (30, 'C', 'Cautiousness', False, 'Avoid mistakes.'),
 (31, 'N', 'Anxiety', False, 'Fear for the worst.'),
 (32, 'E', 'Friendliness', False, 'Warm up quickly to others.'),
 (33, 'O', 'Imagination', False, 'Enjoy wild flights of fantasy.'),
 (34, 'A', 'Trust', False, 'Believe that others have good intentions.'),
 (35, 'C', 'Self-Efficacy', False, 'Excel in what I do.'),
 (36, 'N', 'Anger', False, 'Get irritated easily.'),
 (37, 'E', 'Gregariousness', False, 'Talk to a lot of different people at parties.'),
 (38, 'O', 'Artistic Interests', False, 'Like music.'),
 (39, 'A', 'Morality', False, 'Stick to the rules.'),
 (40, 'C', 'Orderliness', False, 'Like to tidy up.'),
 (41, 'N', 'Depression', False, 'Dislike myself.'),
 (42, 'E', 'Assertiveness', False, 'Try to lead others.'),
 (43, 'O', 'Emotionality', False, "Feel others' emotions."),
 (44, 'A', 'Altruism', False, 'Anticipate the needs of others.'),
 (45, 'C', 'Dutifulness', False, 'Keep my promises.'),
 (46, 'N', 'Self-Consciousness', False, 'Am afraid that I will do the wrong thing.'),
 (47, 'E', 'Activity Level', False, 'Am always on the go.'),
 (48, 'O', 'Adventurousness', False, 'Like to visit new places.'),
 (49, 'A', 'Cooperation', False, "Can't stand confrontations."),
 (50, 'C', 'Achievement-Striving', False, 'Work hard.'),
 (51, 'N', 'Immoderation', False, "Don't know why I do some of the things I do."),
 (52, 'E', 'Excitement-Seeking', False, 'Seek adventure.'),
 (53, 'O', 'Intellect', False, 'Love to read challenging material.'),
 (54, 'A', 'Modesty', False, 'Dislike talking about myself.'),
 (55, 'C', 'Self-Discipline', False, 'Am always prepared.'),
 (56, 'N', 'Vulnerability', False, 'Become overwhelmed by events.'),
 (57, 'E', 'Cheerfulness', False, 'Have a lot of fun.'),
 (58, 'O', 'Liberalism', False, 'Believe that there is no absolute right or wrong.'),
 (59, 'A', 'Sympathy', False, 'Feel sympathy for those who are worse off than myself.'),
 (60, 'C', 'Cautiousness', False, 'Choose my words with care.'),
 (61, 'N', 'Anxiety', False, 'Am afraid of many things.'),
 (62, 'E', 'Friendliness', False, 'Feel comfortable around people.'),
 (63, 'O', 'Imagination', False, 'Love to daydream.'),
 (64, 'A', 'Trust', False, 'Trust what people say.'),
 (65, 'C', 'Self-Efficacy', False, 'Handle tasks smoothly.'),
 (66, 'N', 'Anger', False, 'Get upset easily.'),
 (67, 'E', 'Gregariousness', False, 'Enjoy being part of a group.'),
 (68, 'O', 'Artistic Interests', False, 'See beauty in things that others might not notice.'),
 (69, 'A', 'Morality', True, 'Use flattery to get ahead.'),
 (70, 'C', 'Orderliness', False, 'Want everything to be "just right."'),
 (71, 'N', 'Depression', False, 'Am often down in the dumps.'),
 (72, 'E', 'Assertiveness', False, 'Can talk others into doing things.'),
 (73, 'O', 'Emotionality', False, 'Am passionate about causes.'),
 (74, 'A', 'Altruism', False, 'Love to help others.'),
 (75, 'C', 'Dutifulness', False, 'Pay my bills on time.'),
 (76, 'N', 'Self-Consciousness', False, 'Find it difficult to approach others.'),
 (77, 'E', 'Activity Level', False, 'Do a lot in my spare time.'),
 (78, 'O', 'Adventurousness', False, 'Interested in many things.'),
 (79, 'A', 'Cooperation', False, 'Hate to seem pushy.'),
 (80, 'C', 'Achievement-Striving', False, 'Turn plans into actions.'),
 (81, 'N', 'Immoderation', False, 'Do things I later regret.'),
 (82, 'E', 'Excitement-Seeking', False, 'Love action.'),
 (83, 'O', 'Intellect', False, 'Have a rich vocabulary.'),
 (84, 'A', 'Modesty', False, 'Consider myself an average person.'),
 (85, 'C', 'Self-Discipline', False, 'Start tasks right away.'),
 (86, 'N', 'Vulnerability', False, "Feel that I'm unable to deal with things."),
 (87, 'E', 'Cheerfulness', False, 'Express childlike joy.'),
 (88, 'O', 'Liberalism', False, 'Believe that criminals should receive help rather than punishment.'),
 (89, 'A', 'Sympathy', False, 'Value cooperation over competition.'),
 (90, 'C', 'Cautiousness', False, 'Stick to my chosen path.'),
 (91, 'N', 'Anxiety', False, 'Get stressed out easily.'),
 (92, 'E', 'Friendliness', False, 'Act comfortably with others.'),
 (93, 'O', 'Imagination', False, 'Like to get lost in thought.'),
 (94, 'A', 'Trust', False, 'Believe that people are basically moral.'),
 (95, 'C', 'Self-Efficacy', False, 'Am sure of my ground.'),
 (96, 'N', 'Anger', False, 'Am often in a bad mood.'),
 (97, 'E', 'Gregariousness', False, 'Involve others in what I am doing.'),
 (98, 'O', 'Artistic Interests', False, 'Love flowers.'),
 (99, 'A', 'Morality', True, 'Use others for my own ends.'),
 (100, 'C', 'Orderliness', False, 'Love order and regularity.'),
 (101, 'N', 'Depression', False, 'Have a low opinion of myself.'),
 (102, 'E', 'Assertiveness', False, 'Seek to influence others.'),
 (103, 'O', 'Emotionality', False, 'Enjoy examining myself and my life.'),
 (104, 'A', 'Altruism', False, 'Am concerned about others.'),
 (105, 'C', 'Dutifulness', False, 'Tell the truth.'),
 (106, 'N', 'Self-Consciousness', False, 'Am afraid to draw attention to myself.'),
 (107, 'E', 'Activity Level', False, 'Can manage many things at the same time.'),
 (108, 'O', 'Adventurousness', False, 'Like to begin new things.'),
 (109, 'A', 'Cooperation', True, 'Have a sharp tongue.'),
 (110, 'C', 'Achievement-Striving', False, 'Plunge into tasks with all my heart.'),
 (111, 'N', 'Immoderation', False, 'Go on binges.'),
 (112, 'E', 'Excitement-Seeking', False, 'Enjoy being part of a loud crowd.'),
 (113, 'O', 'Intellect', False, 'Can handle a lot of information.'),
 (114, 'A', 'Modesty', False, 'Seldom toot my own horn.'),
 (115, 'C', 'Self-Discipline', False, 'Get to work at once.'),
 (116, 'N', 'Vulnerability', False, "Can't make up my mind."),
 (117, 'E', 'Cheerfulness', False, 'Laugh my way through life.'),
 (118, 'O', 'Liberalism', True, 'Believe in one true religion.'),
 (119, 'A', 'Sympathy', False, "Suffer from others' sorrows."),
 (120, 'C', 'Cautiousness', True, 'Jump into things without thinking.'),
 (121, 'N', 'Anxiety', False, 'Get caught up in my problems.'),
 (122, 'E', 'Friendliness', False, 'Cheer people up.'),
 (123, 'O', 'Imagination', False, 'Indulge in my fantasies.'),
 (124, 'A', 'Trust', False, 'Believe in human goodness.'),
 (125, 'C', 'Self-Efficacy', False, 'Come up with good solutions.'),
 (126, 'N', 'Anger', False, 'Lose my temper.'),
 (127, 'E', 'Gregariousness', False, 'Love surprise parties.'),
 (128, 'O', 'Artistic Interests', False, 'Enjoy the beauty of nature.'),
 (129, 'A', 'Morality', True, 'Know how to get around the rules.'),
 (130, 'C', 'Orderliness', False, 'Do things according to a plan.'),
 (131, 'N', 'Depression', False, 'Have frequent mood swings.'),
 (132, 'E', 'Assertiveness', False, 'Take control of things.'),
 (133, 'O', 'Emotionality', False, 'Try to understand myself.'),
 (134, 'A', 'Altruism', False, 'Have a good word for everyone.'),
 (135, 'C', 'Dutifulness', False, 'Listen to my conscience.'),
 (136, 'N', 'Self-Consciousness', False, 'Only feel comfortable with friends.'),
 (137, 'E', 'Activity Level', False, 'React quickly.'),
 (138, 'O', 'Adventurousness', True, 'Prefer to stick with things that I know.'),
 (139, 'A', 'Cooperation', True, 'Contradict others.'),
 (140, 'C', 'Achievement-Striving', False, "Do more than what's expected of me."),
 (141, 'N', 'Immoderation', False, 'Love to eat.'),
 (142, 'E', 'Excitement-Seeking', False, 'Enjoy being reckless.'),
 (143, 'O', 'Intellect', False, 'Enjoy thinking about things.'),
 (144, 'A', 'Modesty', True, 'Believe that I am better than others.'),
 (145, 'C', 'Self-Discipline', False, 'Carry out my plans.'),
 (146, 'N', 'Vulnerability', False, 'Get overwhelmed by emotions.'),
 (147, 'E', 'Cheerfulness', False, 'Love life.'),
 (148, 'O', 'Liberalism', True, 'Tend to vote for conservative political candidates.'),
 (149, 'A', 'Sympathy', True, "Am not interested in other people's problems."),
 (150, 'C', 'Cautiousness', True, 'Make rash decisions.'),
 (151, 'N', 'Anxiety', True, 'Am not easily bothered by things.'),
 (152, 'E', 'Friendliness', True, 'Am hard to get to know.'),
 (153, 'O', 'Imagination', False, 'Spend time reflecting on things.'),
 (154, 'A', 'Trust', False, 'Think that all will be well.'),
 (155, 'C', 'Self-Efficacy', False, 'Know how to get things done.'),
 (156, 'N', 'Anger', True, 'Rarely get irritated.'),
 (157, 'E', 'Gregariousness', True, 'Prefer to be alone.'),
 (158, 'O', 'Artistic Interests', True, 'Do not like art.'),
 (159, 'A', 'Morality', True, 'Cheat to get ahead.'),
 (160, 'C', 'Orderliness', True, 'Often forget to put things back in their proper place.'),
 (161, 'N', 'Depression', False, 'Feel desperate.'),
 (162, 'E', 'Assertiveness', True, 'Wait for others to lead the way.'),
 (163, 'O', 'Emotionality', True, 'Seldom get emotional.'),
 (164, 'A', 'Altruism', True, 'Look down on others.'),
 (165, 'C', 'Dutifulness', True, 'Break rules.'),
 (166, 'N', 'Self-Consciousness', False, 'Stumble over my words.'),
 (167, 'E', 'Activity Level', True, 'Like to take it easy.'),
 (168, 'O', 'Adventurousness', True, 'Dislike changes.'),
 (169, 'A', 'Cooperation', True, 'Love a good fight.'),
 (170, 'C', 'Achievement-Striving', False, 'Set high standards for myself and others.'),
 (171, 'N', 'Immoderation', True, 'Rarely overindulge.'),
 (172, 'E', 'Excitement-Seeking', False, 'Act wild and crazy.'),
 (173, 'O', 'Intellect', True, 'Am not interested in abstract ideas.'),
 (174, 'A', 'Modesty', True, 'Think highly of myself.'),
 (175, 'C', 'Self-Discipline', True, 'Find it difficult to get down to work.'),
 (176, 'N', 'Vulnerability', True, 'Remain calm under pressure.'),
 (177, 'E', 'Cheerfulness', False, 'Look at the bright side of life.'),
 (178, 'O', 'Liberalism', True, 'Believe that too much tax money goes to support artists.'),
 (179, 'A', 'Sympathy', True, 'Tend to dislike soft-hearted people.'),
 (180, 'C', 'Cautiousness', True, 'Like to act on a whim.'),
 (181, 'N', 'Anxiety', True, 'Am relaxed most of the time.'),
 (182, 'E', 'Friendliness', True, 'Often feel uncomfortable around others.'),
 (183, 'O', 'Imagination', True, 'Seldom daydream.'),
 (184, 'A', 'Trust', True, 'Distrust people.'),
 (185, 'C', 'Self-Efficacy', True, 'Misjudge situations.'),
 (186, 'N', 'Anger', True, 'Seldom get mad.'),
 (187, 'E', 'Gregariousness', True, 'Want to be left alone.'),
 (188, 'O', 'Artistic Interests', True, 'Do not like poetry.'),
 (189, 'A', 'Morality', True, 'Put people under pressure.'),
 (190, 'C', 'Orderliness', True, 'Leave a mess in my room.'),
 (191, 'N', 'Depression', False, 'Feel that my life lacks direction.'),
 (192, 'E', 'Assertiveness', True, 'Keep in the background.'),
 (193, 'O', 'Emotionality', True, 'Am not easily affected by my emotions.'),
 (194, 'A', 'Altruism', True, 'Am indifferent to the feelings of others.'),
 (195, 'C', 'Dutifulness', True, 'Break my promises.'),
 (196, 'N', 'Self-Consciousness', True, 'Am not embarrassed easily.'),
 (197, 'E', 'Activity Level', True, 'Like to take my time.'),
 (198, 'O', 'Adventurousness', True, "Don't like the idea of change."),
 (199, 'A', 'Cooperation', True, 'Yell at people.'),
 (200, 'C', 'Achievement-Striving', False, 'Demand quality.'),
 (201, 'N', 'Immoderation', True, 'Easily resist temptations.'),
 (202, 'E', 'Excitement-Seeking', False, 'Willing to try anything once.'),
 (203, 'O', 'Intellect', True, 'Avoid philosophical discussions.'),
 (204, 'A', 'Modesty', True, 'Have a high opinion of myself.'),
 (205, 'C', 'Self-Discipline', True, 'Waste my time.'),
 (206, 'N', 'Vulnerability', True, 'Can handle complex problems.'),
 (207, 'E', 'Cheerfulness', False, 'Laugh aloud.'),
 (208, 'O', 'Liberalism', True, 'Believe laws should be strictly enforced.'),
 (209, 'A', 'Sympathy', True, 'Believe in an eye for an eye.'),
 (210, 'C', 'Cautiousness', True, 'Rush into things.'),
 (211, 'N', 'Anxiety', True, 'Am not easily disturbed by events.'),
 (212, 'E', 'Friendliness', True, 'Avoid contacts with others.'),
 (213, 'O', 'Imagination', True, 'Do not have a good imagination.'),
 (214, 'A', 'Trust', True, 'Suspect hidden motives in others.'),
 (215, 'C', 'Self-Efficacy', True, "Don't understand things."),
 (216, 'N', 'Anger', True, 'Am not easily annoyed.'),
 (217, 'E', 'Gregariousness', True, "Don't like crowded events."),
 (218, 'O', 'Artistic Interests', True, 'Do not enjoy going to art museums.'),
 (219, 'A', 'Morality', True, 'Pretend to be concerned for others.'),
 (220, 'C', 'Orderliness', True, 'Leave my belongings around.'),
 (221, 'N', 'Depression', True, 'Seldom feel blue.'),
 (222, 'E', 'Assertiveness', True, 'Have little to say.'),
 (223, 'O', 'Emotionality', True, 'Rarely notice my emotional reactions.'),
 (224, 'A', 'Altruism', True, 'Make people feel uncomfortable.'),
 (225, 'C', 'Dutifulness', True, 'Get others to do my duties.'),
 (226, 'N', 'Self-Consciousness', True, 'Am comfortable in unfamiliar situations.'),
 (227, 'E', 'Activity Level', True, 'Like a leisurely lifestyle.'),
 (228, 'O', 'Adventurousness', True, 'Am a creature of habit.'),
 (229, 'A', 'Cooperation', True, 'Insult people.'),
 (230, 'C', 'Achievement-Striving', True, 'Am not highly motivated to succeed.'),
 (231, 'N', 'Immoderation', True, 'Am able to control my cravings.'),
 (232, 'E', 'Excitement-Seeking', False, 'Seek danger.'),
 (233, 'O', 'Intellect', True, 'Have difficulty understanding abstract ideas.'),
 (234, 'A', 'Modesty', True, 'Know the answers to many questions.'),
 (235, 'C', 'Self-Discipline', True, 'Need a push to get started.'),
 (236, 'N', 'Vulnerability', True, 'Know how to cope.'),
 (237, 'E', 'Cheerfulness', False, 'Amuse my friends.'),
 (238, 'O', 'Liberalism', True, 'Believe that we coddle criminals too much.'),
 (239, 'A', 'Sympathy', True, 'Try not to think about the needy.'),
 (240, 'C', 'Cautiousness', True, 'Do crazy things.'),
 (241, 'N', 'Anxiety', True, "Don't worry about things that have already happened."),
 (242, 'E', 'Friendliness', True, 'Am not really interested in others.'),
 (243, 'O', 'Imagination', True, 'Seldom get lost in thought.'),
 (244, 'A', 'Trust', True, 'Am wary of others.'),
 (245, 'C', 'Self-Efficacy', True, 'Have little to contribute.'),
 (246, 'N', 'Anger', True, 'Keep my cool.'),
 (247, 'E', 'Gregariousness', True, 'Avoid crowds.'),
 (248, 'O', 'Artistic Interests', True, 'Do not like concerts.'),
 (249, 'A', 'Morality', True, 'Take advantage of others.'),
 (250, 'C', 'Orderliness', True, 'Am not bothered by messy people.'),
 (251, 'N', 'Depression', True, 'Feel comfortable with myself.'),
 (252, 'E', 'Assertiveness', True, "Don't like to draw attention to myself."),
 (253, 'O', 'Emotionality', True, 'Experience very few emotional highs and lows.'),
 (254, 'A', 'Altruism', True, 'Turn my back on others.'),
 (255, 'C', 'Dutifulness', True, 'Do the opposite of what is asked.'),
 (256, 'N', 'Self-Consciousness', True, 'Am not bothered by difficult social situations.'),
 (257, 'E', 'Activity Level', True, 'Let things proceed at their own pace.'),
 (258, 'O', 'Adventurousness', True, 'Dislike new foods.'),
 (259, 'A', 'Cooperation', True, 'Get back at others.'),
 (260, 'C', 'Achievement-Striving', True, 'Do just enough work to get by.'),
 (261, 'N', 'Immoderation', True, 'Never spend more than I can afford.'),
 (262, 'E', 'Excitement-Seeking', True, 'Would never go hang gliding or bungee jumping.'),
 (263, 'O', 'Intellect', True, 'Am not interested in theoretical discussions.'),
 (264, 'A', 'Modesty', True, 'Boast about my virtues.'),
 (265, 'C', 'Self-Discipline', True, 'Have difficulty starting tasks.'),
 (266, 'N', 'Vulnerability', True, 'Readily overcome setbacks.'),
 (267, 'E', 'Cheerfulness', True, 'Am not easily amused.'),
 (268, 'O', 'Liberalism', True, 'Believe that we should be tough on crime.'),
 (269, 'A', 'Sympathy', True, 'Believe people should fend for themselves.'),
 (270, 'C', 'Cautiousness', True, 'Act without thinking.'),
 (271, 'N', 'Anxiety', True, 'Adapt easily to new situations.'),
 (272, 'E', 'Friendliness', True, 'Keep others at a distance.'),
 (273, 'O', 'Imagination', True, 'Have difficulty imagining things.'),
 (274, 'A', 'Trust', True, 'Believe that people are essentially evil.'),
 (275, 'C', 'Self-Efficacy', True, "Don't see the consequences of things."),
 (276, 'N', 'Anger', True, 'Rarely complain.'),
 (277, 'E', 'Gregariousness', True, 'Seek quiet.'),
 (278, 'O', 'Artistic Interests', True, 'Do not enjoy watching dance performances.'),
 (279, 'A', 'Morality', True, "Obstruct others' plans."),
 (280, 'C', 'Orderliness', True, 'Am not bothered by disorder.'),
 (281, 'N', 'Depression', True, 'Am very pleased with myself.'),
 (282, 'E', 'Assertiveness', True, 'Hold back my opinions.'),
 (283, 'O', 'Emotionality', True, "Don't understand people who get emotional."),
 (284, 'A', 'Altruism', True, 'Take no time for others.'),
 (285, 'C', 'Dutifulness', True, 'Misrepresent the facts.'),
 (286, 'N', 'Self-Consciousness', True, 'Am able to stand up for myself.'),
 (287, 'E', 'Activity Level', True, 'React slowly.'),
 (288, 'O', 'Adventurousness', True, 'Am attached to conventional ways.'),
 (289, 'A', 'Cooperation', True, 'Hold a grudge.'),
 (290, 'C', 'Achievement-Striving', True, 'Put little time and effort into my work.'),
 (291, 'N', 'Immoderation', True, 'Never splurge.'),
 (292, 'E', 'Excitement-Seeking', True, 'Dislike loud music.'),
 (293, 'O', 'Intellect', True, 'Avoid difficult reading material.'),
 (294, 'A', 'Modesty', True, 'Make myself the center of attention.'),
 (295, 'C', 'Self-Discipline', True, 'Postpone decisions.'),
 (296, 'N', 'Vulnerability', True, 'Am calm even in tense situations.'),
 (297, 'E', 'Cheerfulness', True, 'Seldom joke around.'),
 (298, 'O', 'Liberalism', True, 'Like to stand during the national anthem.'),
 (299, 'A', 'Sympathy', True, "Can't stand weak people."),
 (300, 'C', 'Cautiousness', True, 'Often make last-minute plans.')]

ITEM_TO_DOMAIN = {item_num: domain for item_num, domain, facet, reverse, item_text in ITEM_KEY_300}
ITEM_TO_FACET = {item_num: facet for item_num, domain, facet, reverse, item_text in ITEM_KEY_300}
ITEM_TO_REVERSE = {item_num: reverse for item_num, domain, facet, reverse, item_text in ITEM_KEY_300}
ITEM_TO_TEXT = {item_num: item_text for item_num, domain, facet, reverse, item_text in ITEM_KEY_300}
FACET_ORDER = []
FACET_TO_DOMAIN = {}
for item_num, domain, facet, reverse, item_text in ITEM_KEY_300:
    if facet not in FACET_ORDER:
        FACET_ORDER.append(facet)
    FACET_TO_DOMAIN[facet] = domain

TEST_KEY = [entry for entry in ITEM_KEY_300 if 121 <= entry[0] <= 300]
TRAIN_KEY = [entry for entry in ITEM_KEY_300 if 1 <= entry[0] <= 120]
TEST_DOMAIN_COUNTS = {d: sum(1 for item_num, domain, facet, reverse, text in TEST_KEY if domain == d) for d in DOMAIN_ORDER}
TRAIN_DOMAIN_COUNTS = {d: sum(1 for item_num, domain, facet, reverse, text in TRAIN_KEY if domain == d) for d in DOMAIN_ORDER}
FULL_DOMAIN_COUNTS = {d: sum(1 for item_num, domain, facet, reverse, text in ITEM_KEY_300 if domain == d) for d in DOMAIN_ORDER}
FULL_FACET_COUNTS = {f: sum(1 for item_num, domain, facet, reverse, text in ITEM_KEY_300 if facet == f) for f in FACET_ORDER}

assert len(ITEM_KEY_300) == 300, "ITEM_KEY_300 must contain exactly 300 items."


# =============================================================================
# Model registry
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    provider: str
    model_id: str
    notes: str


AVAILABLE_MODELS: Dict[str, ModelSpec] = {
    "gpt-5.4": ModelSpec(
        key="gpt-5.4",
        display="GPT-5.4",
        provider="openai_responses",
        model_id="gpt-5.4",
        notes="OpenAI Responses API",
    ),
    "gpt-5.4-mini": ModelSpec(
        key="gpt-5.4-mini",
        display="GPT-5.4 mini",
        provider="openai_responses",
        model_id="gpt-5.4-mini",
        notes="OpenAI Responses API",
    ),
    "gpt-5.4-nano": ModelSpec(
        key="gpt-5.4-nano",
        display="GPT-5.4 nano",
        provider="openai_responses",
        model_id="gpt-5.4-nano",
        notes="OpenAI Responses API",
    ),
    "gpt-realtime-1.5": ModelSpec(
        key="gpt-realtime-1.5",
        display="GPT-Realtime-1.5",
        provider="openai_realtime",
        model_id="gpt-realtime-1.5",
        notes="OpenAI Realtime API, text-only validation mode",
    ),
    "gemma-4-26b-a4b-it": ModelSpec(
        key="gemma-4-26b-a4b-it",
        display="Gemma 4 26B A4B IT",
        provider="google_genai",
        model_id="gemma-4-26b-a4b-it",
        notes="Google AI Studio / Gemini API via google-genai SDK",
    ),
    "gemma-4-31b-it": ModelSpec(
        key="gemma-4-31b-it",
        display="Gemma 4 31B IT",
        provider="google_genai",
        model_id="gemma-4-31b-it",
        notes="Google AI Studio / Gemini API via google-genai SDK",
    ),
    "llama-3.1-8b-instant": ModelSpec(
        key="llama-3.1-8b-instant",
        display="Llama 3.1 8B Instant 128k",
        provider="groq_chat",
        model_id="llama-3.1-8b-instant",
        notes="Groq Chat Completions",
    ),
}

MIGRATED_DEFAULT_SELECTION = {
    "instructors": ["llama-3.1-8b-instant"],
    "targets": ["gpt-5.4-nano"],
}


# =============================================================================
# Logging and caching
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_model_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s")
    file_handler = logging.FileHandler(str(output_dir / "multi_model_validation.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_").lower()


def load_json_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


INSTRUCTOR_CACHE_FILE = CACHE_DIR / "instructor_prompt_cache.json"
SYSTEM_PROMPT_CACHE_FILE = CACHE_DIR / "validation_system_prompt_cache.json"
TARGET_BATCH_CACHE_FILE = CACHE_DIR / "target_batch_scores_cache.json"
RAW_TEXT_CACHE_FILE = CACHE_DIR / "raw_text_generation_cache.json"

INSTRUCTOR_CACHE = load_json_cache(INSTRUCTOR_CACHE_FILE)
SYSTEM_PROMPT_CACHE = load_json_cache(SYSTEM_PROMPT_CACHE_FILE)
TARGET_BATCH_CACHE = load_json_cache(TARGET_BATCH_CACHE_FILE)
RAW_TEXT_CACHE = load_json_cache(RAW_TEXT_CACHE_FILE)


# =============================================================================
# Selection menu
# =============================================================================

def load_previous_selection() -> Dict[str, List[str]]:
    if SELECTION_FILE.exists():
        try:
            data = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
            instructors = [m for m in data.get("instructors", []) if m in AVAILABLE_MODELS]
            targets = [m for m in data.get("targets", []) if m in AVAILABLE_MODELS]
            if instructors and targets:
                return {"instructors": instructors, "targets": targets}
        except Exception:
            pass
    return MIGRATED_DEFAULT_SELECTION.copy()


def save_selection(selection: Dict[str, List[str]]) -> None:
    SELECTION_FILE.write_text(json.dumps(selection, indent=2), encoding="utf-8")


def print_model_menu() -> None:
    print("\nAvailable models for BOTH instructor and target roles:")
    for idx, key in enumerate(AVAILABLE_MODELS.keys(), start=1):
        spec = AVAILABLE_MODELS[key]
        print(f"  {idx}. {spec.key:<24} | {spec.display:<24} | {spec.notes}")


def format_model_list(model_keys: Sequence[str]) -> str:
    return ", ".join(model_keys)


def parse_model_selection(raw: str) -> List[str]:
    raw = raw.strip()
    keys = list(AVAILABLE_MODELS.keys())
    if raw.lower() in {"all", "*"}:
        return keys
    selected: List[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if not (1 <= idx <= len(keys)):
                raise ValueError(f"Model number {idx} is out of range.")
            selected.append(keys[idx - 1])
        else:
            token = token.lower()
            if token not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {token}")
            selected.append(token)
    deduped = []
    for m in selected:
        if m not in deduped:
            deduped.append(m)
    if not deduped:
        raise ValueError("No valid models selected.")
    return deduped


def choose_models_interactively() -> Dict[str, List[str]]:
    selection = load_previous_selection()

    print("\n" + "=" * 86)
    print("MODEL SELECTION — no participant data is sent before this step")
    print("=" * 86)
    print("Current chosen instructor model(s):", format_model_list(selection["instructors"]))
    print("Current chosen target model(s)    :", format_model_list(selection["targets"]))
    print("\nPress 1 to continue with the same choice")
    print("Press 2 to change choice")
    choice = input("Choice: ").strip() or "1"

    if choice == "1":
        print("Using previous/current model choice.")
        return selection

    if choice != "2":
        print("Unrecognised choice. Using previous/current model choice.")
        return selection

    print_model_menu()
    print("\nEnter comma-separated numbers or model IDs. Type 'all' to run every model.")
    while True:
        try:
            instructor_raw = input("Instructor model(s): ").strip()
            instructors = parse_model_selection(instructor_raw)
            target_raw = input("Target model(s): ").strip()
            targets = parse_model_selection(target_raw)
            new_selection = {"instructors": instructors, "targets": targets}
            save_selection(new_selection)
            print("\nSaved model choice.")
            print("Instructor model(s):", format_model_list(instructors))
            print("Target model(s)    :", format_model_list(targets))
            return new_selection
        except ValueError as e:
            print(f"Selection error: {e}")
            print("Try again.")


def ask_max_participants(default: int) -> Optional[int]:
    raw = input(f"How many participants to validate? Press Enter for {default}, or type 'all': ").strip()
    if not raw:
        return default
    if raw.lower() == "all":
        return None
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Invalid participant count. Using {default}.")
        return default


# =============================================================================
# Environment validation and lazy clients
# =============================================================================

def selected_providers(selection: Dict[str, List[str]]) -> set:
    providers = set()
    for key in selection["instructors"] + selection["targets"]:
        providers.add(AVAILABLE_MODELS[key].provider)
    return providers


def validate_environment(selection: Dict[str, List[str]]) -> None:
    providers = selected_providers(selection)
    missing = []
    if "openai_responses" in providers or "openai_realtime" in providers:
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
    if "groq_chat" in providers:
        if not GROQ_API_KEY:
            missing.append("GROQ_API_KEY or GROQ_API_KEY_TUD")
    if "google_genai" in providers:
        if not GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY or GOOGLE_AI_STUDIO_API_KEY or GOOGLE_API_KEY")
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))

    if "google_genai" in providers:
        print("\nGemma selected.")
        print("  Provider: Google AI Studio / Gemini API")
        print(f"  GEMINI_API_KEY / GOOGLE_AI_STUDIO_API_KEY / GOOGLE_API_KEY = {'set' if GEMINI_API_KEY else 'not set'}")
        print("  Install dependency if needed: python -m pip install -U google-genai\n")


_OPENAI_CLIENT = None
_ASYNC_OPENAI_CLIENT = None
_GROQ_CLIENT = None
_GOOGLE_GENAI_CLIENT = None


def get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
    return _OPENAI_CLIENT


def get_async_openai_client():
    global _ASYNC_OPENAI_CLIENT
    if _ASYNC_OPENAI_CLIENT is None:
        from openai import AsyncOpenAI
        _ASYNC_OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=0)
    return _ASYNC_OPENAI_CLIENT


def get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        from groq import Groq
        _GROQ_CLIENT = Groq(api_key=GROQ_API_KEY, max_retries=0)
    return _GROQ_CLIENT


def get_google_genai_client():
    global _GOOGLE_GENAI_CLIENT
    if _GOOGLE_GENAI_CLIENT is None:
        from google import genai
        _GOOGLE_GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return _GOOGLE_GENAI_CLIENT


# =============================================================================
# Retry and parsing helpers
# =============================================================================

def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("429" in text) or ("rate limit" in text) or ("too many requests" in text)


def is_temporary_server_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(phrase in text for phrase in [
        "500", "502", "503", "504",
        "internal server error", "bad gateway", "service unavailable",
        "gateway timeout", "connection reset", "timed out", "timeout",
        "connection error", "server disconnected",
    ])


def safe_get_header(exc: Exception, key: str) -> Optional[str]:
    try:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            return headers.get(key)
    except Exception:
        return None
    return None


def compute_retry_delay(exc: Exception, attempt: int, default_base: float = 3.0, default_cap: float = 120.0) -> float:
    retry_after = safe_get_header(exc, "retry-after")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    exponential = min(default_base * (2 ** attempt), default_cap)
    return exponential + random.uniform(0, 1.5)


def extract_json_candidate(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    obj_start, obj_end = cleaned.find("{"), cleaned.rfind("}")
    arr_start, arr_end = cleaned.find("["), cleaned.rfind("]")
    candidates = []
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(cleaned[obj_start:obj_end + 1])
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(cleaned[arr_start:arr_end + 1])
    if not candidates:
        raise ValueError(f"No JSON object or array found in output: {cleaned[:250]}")
    # Prefer object because OpenAI structured output returns {"scores": [...]}.
    candidates.sort(key=lambda c: 0 if c.startswith("{") else 1)
    return candidates[0]


def parse_scores_json(raw_text: str, expected_n: int) -> List[int]:
    snippet = extract_json_candidate(raw_text)
    parsed = json.loads(snippet)
    if isinstance(parsed, dict):
        scores = parsed.get("scores")
    elif isinstance(parsed, list):
        scores = parsed
    else:
        raise ValueError("JSON output must be a scores object or array.")

    if not isinstance(scores, list):
        raise ValueError(f"JSON output missing scores list: {str(parsed)[:250]}")
    if len(scores) != expected_n:
        raise ValueError(f"Expected {expected_n} scores, got {len(scores)}.")

    clean_scores: List[int] = []
    for value in scores:
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"Invalid score value: {value} in {scores}")
        clean_scores.append(value)
    return clean_scores


def extract_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return direct.strip()

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            ctype = getattr(content, "type", None)
            if ctype in {"output_text", "text"}:
                piece = getattr(content, "text", None)
                if piece:
                    chunks.append(piece)
    return "".join(chunks).strip()


def build_scores_schema(n: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": f"personality_scores_{n}",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 5},
                    "minItems": n,
                    "maxItems": n,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    }


def log_openai_prompt_cache_usage(response: Any, logger: logging.Logger, label: str) -> None:
    try:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "input_tokens_details", None) if usage else None
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        cached_tokens = getattr(details, "cached_tokens", None) if details else None
        if input_tokens is not None and cached_tokens is not None:
            logger.info(f"    OpenAI prompt cache {label}: cached_tokens={cached_tokens} / input_tokens={input_tokens}")
    except Exception:
        pass


# =============================================================================
# Model call layer
# =============================================================================

def call_model_text(
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    logger: logging.Logger,
    purpose: str,
    max_tokens: int,
    expected_scores: Optional[int] = None,
) -> str:
    cache_key = stable_hash(
        json.dumps({
            "prompt_version": PROMPT_VERSION,
            "model_key": spec.key,
            "provider": spec.provider,
            "purpose": purpose,
            "expected_scores": expected_scores,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }, ensure_ascii=False, sort_keys=True)
    )
    # Cache raw free-text generations such as instructor prompts.
    # For score batches, cache only after successful JSON parsing in TARGET_BATCH_CACHE;
    # otherwise a malformed raw response could poison future runs.
    if expected_scores is None:
        cached = RAW_TEXT_CACHE.get(cache_key)
        if cached is not None:
            logger.info(f"    raw text cache hit: {purpose} | {spec.key}")
            return cached

    for attempt in range(8):
        try:
            if spec.provider == "openai_responses":
                text = call_openai_responses(spec, system_prompt, user_prompt, logger, purpose, max_tokens, expected_scores)
            elif spec.provider == "openai_realtime":
                text = asyncio.run(call_openai_realtime(spec, system_prompt, user_prompt, logger, purpose, max_tokens))
            elif spec.provider == "groq_chat":
                text = call_groq_chat(spec, system_prompt, user_prompt, logger, purpose, max_tokens, expected_scores)
            elif spec.provider == "google_genai":
                text = call_google_genai(spec, system_prompt, user_prompt, logger, purpose, max_tokens, expected_scores)
            else:
                raise RuntimeError(f"Unsupported provider: {spec.provider}")

            if expected_scores is None:
                RAW_TEXT_CACHE[cache_key] = text
                save_json_cache(RAW_TEXT_CACHE_FILE, RAW_TEXT_CACHE)
            return text

        except Exception as e:
            if is_rate_limit_error(e) or is_temporary_server_error(e):
                delay = compute_retry_delay(e, attempt)
                logger.warning(f"    retry {attempt + 1} for {purpose} | {spec.key} after error: {repr(e)} | waiting {delay:.1f}s")
                time.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Failed after repeated retries: {purpose} | {spec.key}")


def call_openai_responses(
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    logger: logging.Logger,
    purpose: str,
    max_tokens: int,
    expected_scores: Optional[int],
) -> str:
    client = get_openai_client()
    kwargs: Dict[str, Any] = {
        "model": spec.model_id,
        "temperature": 0,
        "max_output_tokens": max_tokens,
        "prompt_cache_key": stable_hash(system_prompt),
        "prompt_cache_retention": "24h",
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    }
    if DEFAULT_REASONING_EFFORT.lower() not in {"", "none-disabled", "disabled"}:
        kwargs["reasoning"] = {"effort": DEFAULT_REASONING_EFFORT}
    if expected_scores is not None:
        kwargs["text"] = {"format": build_scores_schema(expected_scores)}

    response = client.responses.create(**kwargs)
    log_openai_prompt_cache_usage(response, logger, purpose)
    text = extract_response_text(response)
    if not text:
        raise RuntimeError("OpenAI Responses API returned empty text.")
    return text


async def call_openai_realtime(
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    logger: logging.Logger,
    purpose: str,
    max_tokens: int,
) -> str:
    client = get_async_openai_client()
    async with client.realtime.connect(model=spec.model_id) as connection:
        await connection.session.update(
            session={
                "type": "realtime",
                "output_modalities": ["text"],
            }
        )

        await connection.response.create(
            response={
                "conversation": "none",
                "output_modalities": ["text"],
                "max_output_tokens": max_tokens,
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
            }
        )

        deltas: List[str] = []
        final_text: Optional[str] = None
        async for event in connection:
            etype = getattr(event, "type", None)
            if etype == "response.output_text.delta":
                deltas.append(getattr(event, "delta", "") or "")
            elif etype == "response.output_text.done":
                text_value = getattr(event, "text", None)
                if text_value:
                    final_text = text_value
            elif etype == "error":
                err = getattr(event, "error", None)
                message = getattr(err, "message", "Unknown Realtime API error")
                code = getattr(err, "code", None)
                raise RuntimeError(f"Realtime API error{f' [{code}]' if code else ''}: {message}")
            elif etype == "response.done":
                if final_text:
                    return final_text.strip()
                combined = "".join(deltas).strip()
                if combined:
                    return combined
                # last-resort event dump parser
                try:
                    payload = event.model_dump() if hasattr(event, "model_dump") else {}
                    chunks = []
                    for item in (payload.get("response") or {}).get("output") or []:
                        for content in item.get("content") or []:
                            if content.get("type") in {"output_text", "text"} and content.get("text"):
                                chunks.append(content["text"])
                    fallback = "".join(chunks).strip()
                    if fallback:
                        return fallback
                except Exception:
                    pass
                raise RuntimeError("Realtime response completed without text output.")
    raise RuntimeError("Realtime connection closed before response.done.")


def call_groq_chat(
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    logger: logging.Logger,
    purpose: str,
    max_tokens: int,
    expected_scores: Optional[int],
) -> str:
    client = get_groq_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: Dict[str, Any] = {
        "model": spec.model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if expected_scores is not None:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some OpenAI-compatible and Groq account configs may reject response_format.
        if expected_scores is not None and "response_format" in str(e).lower():
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("Groq returned empty text.")
    return text.strip()


def build_gemini_scores_schema(n: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 5},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def call_google_genai(
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    logger: logging.Logger,
    purpose: str,
    max_tokens: int,
    expected_scores: Optional[int],
) -> str:
    """Call Gemma through Google AI Studio / Gemini API.

    This replaces the earlier local OpenAI-compatible Gemma path. It uses the
    google-genai SDK with system_instruction and contents, matching the hosted
    Google AI Studio flow rather than a workstation server.
    """
    client = get_google_genai_client()
    from google.genai import types

    base_config_kwargs: Dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": 0,
        "max_output_tokens": max_tokens,
    }

    if expected_scores is not None:
        # JSON mode for score batches. The prompt still asks for {"scores": [...]}
        # and parse_scores_json verifies length/range before caching.
        base_config_kwargs["response_mime_type"] = "application/json"
        base_config_kwargs["response_schema"] = build_gemini_scores_schema(expected_scores)

    def _generate(config_kwargs: Dict[str, Any]) -> Any:
        config = types.GenerateContentConfig(**config_kwargs)
        return client.models.generate_content(
            model=spec.model_id,
            contents=user_prompt,
            config=config,
        )

    try:
        response = _generate(dict(base_config_kwargs))
    except Exception as e:
        # Some google-genai versions or individual models may reject response_schema.
        # Fall back to JSON MIME mode without schema; parse_scores_json still validates.
        message = str(e).lower()
        if expected_scores is not None and (
            "response_schema" in message
            or "responseschema" in message
            or "schema" in message
            or "unknown field" in message
            or "unsupported" in message
        ):
            logger.warning("    Google GenAI schema config rejected; retrying with JSON MIME only.")
            fallback_kwargs = dict(base_config_kwargs)
            fallback_kwargs.pop("response_schema", None)
            response = _generate(fallback_kwargs)
        else:
            raise

    text = getattr(response, "text", None)
    if not text:
        try:
            chunks = []
            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        chunks.append(part_text)
            text = "".join(chunks).strip()
        except Exception:
            text = ""

    if not text:
        raise RuntimeError("Google GenAI returned empty text.")
    return text.strip()


# =============================================================================
# Scoring and prompt building
# =============================================================================

def normalize_raw_score(value: Any) -> int:
    try:
        raw = int(value)
    except Exception:
        raw = 3
    return raw if 1 <= raw <= 5 else 3


def score_item(raw_score: Any, reverse: bool) -> int:
    raw = normalize_raw_score(raw_score)
    return 6 - raw if reverse else raw


def facet_level_label(value: float) -> str:
    if value < 1.5:
        return "very low"
    if value < 2.5:
        return "low"
    if value < 3.5:
        return "mixed / moderate"
    if value < 4.5:
        return "high"
    return "very high"


def get_scores(row: pd.Series, items: Sequence[int]) -> List[int]:
    return [normalize_raw_score(row.get(f"i{item}", 3)) for item in items]


def summarize_scale_distribution(scores: Sequence[int]) -> Dict[str, Any]:
    clean = [normalize_raw_score(s) for s in scores]
    counts = {score: clean.count(score) for score in range(1, 6)}
    total = len(clean) if clean else 1
    return {
        "mean": round(float(np.mean(clean)), 4),
        "variance": round(float(np.var(clean)), 4),
        "neutral_pct": round(counts[3] / total * 100, 2),
        "extreme_pct": round((counts[1] + counts[5]) / total * 100, 2),
        "low_pct": round((counts[1] + counts[2]) / total * 100, 2),
        "high_pct": round((counts[4] + counts[5]) / total * 100, 2),
        "counts": counts,
    }


def compute_domain_avgs_from_row(row: pd.Series, key_entries: Sequence[Tuple[int, str, str, bool, str]]) -> Dict[str, float]:
    totals = {domain: 0 for domain in DOMAIN_ORDER}
    counts = {domain: 0 for domain in DOMAIN_ORDER}
    for item_num, domain, facet, reverse, item_text in key_entries:
        totals[domain] += score_item(row.get(f"i{item_num}", 3), reverse)
        counts[domain] += 1
    return {domain: round(totals[domain] / counts[domain], 4) for domain in DOMAIN_ORDER}


def compute_domain_avgs_from_scores(scores: Sequence[int], key_entries: Sequence[Tuple[int, str, str, bool, str]], base_item: int) -> Dict[str, float]:
    totals = {domain: 0 for domain in DOMAIN_ORDER}
    counts = {domain: 0 for domain in DOMAIN_ORDER}
    for item_num, domain, facet, reverse, item_text in key_entries:
        idx = item_num - base_item
        totals[domain] += score_item(scores[idx], reverse)
        counts[domain] += 1
    return {domain: round(totals[domain] / counts[domain], 4) for domain in DOMAIN_ORDER}


def compute_facet_avgs_from_row(row: pd.Series) -> Dict[str, float]:
    totals = {facet: 0 for facet in FACET_ORDER}
    counts = {facet: 0 for facet in FACET_ORDER}
    for item_num, domain, facet, reverse, item_text in ITEM_KEY_300:
        totals[facet] += score_item(row.get(f"i{item_num}", 3), reverse)
        counts[facet] += 1
    return {facet: round(totals[facet] / counts[facet], 4) for facet in FACET_ORDER}


def build_profile_summary(row: pd.Series) -> Dict[str, Any]:
    full_scores = get_scores(row, ALL_ITEMS)
    dist = summarize_scale_distribution(full_scores)
    ocean_avgs = compute_domain_avgs_from_row(row, ITEM_KEY_300)
    facet_avgs = compute_facet_avgs_from_row(row)

    ordered_facets = sorted(facet_avgs.items(), key=lambda kv: (kv[1], kv[0]))
    lowest = ordered_facets[:6]
    highest = list(reversed(ordered_facets[-6:]))

    distribution_lines = [f"- score {score}: {dist['counts'][score]} of 300" for score in range(1, 6)]
    ocean_lines = [f"- {DOMAIN_FULL_NAMES[d]} ({d}): avg={ocean_avgs[d]}" for d in DOMAIN_ORDER]
    facet_lines = [
        f"- {facet} ({FACET_TO_DOMAIN[facet]}): avg={facet_avgs[facet]} ({facet_level_label(facet_avgs[facet])})"
        for facet in FACET_ORDER
    ]
    highest_lines = [f"- {facet}: {score}" for facet, score in highest]
    lowest_lines = [f"- {facet}: {score}" for facet, score in lowest]

    summary_text = "\n".join([
        "PROFILE SUMMARY FROM ITEMS 1-300",
        f"- overall mean raw response: {dist['mean']}",
        f"- raw response variance: {dist['variance']}",
        f"- neutral (3) usage: {dist['neutral_pct']}%",
        f"- extreme (1 or 5) usage: {dist['extreme_pct']}%",
        f"- low-end raw response (1 or 2) usage: {dist['low_pct']}%",
        f"- high-end raw response (4 or 5) usage: {dist['high_pct']}%",
        "- raw score distribution:",
        *distribution_lines,
        "- OCEAN averages from all 300 items after reverse scoring where required:",
        *ocean_lines,
        "- highest scoring facets after reverse scoring:",
        *highest_lines,
        "- lowest scoring facets after reverse scoring:",
        *lowest_lines,
        "- all facet averages after reverse scoring:",
        *facet_lines,
    ])

    return {
        "summary_text": summary_text,
        "ocean_avgs": ocean_avgs,
        "facet_avgs": facet_avgs,
        "highest_facets": highest,
        "lowest_facets": lowest,
    }


def build_instructor_evidence(row: pd.Series) -> str:
    profile = build_profile_summary(row)
    facet_blocks: List[str] = []

    for facet in FACET_ORDER:
        domain = FACET_TO_DOMAIN[facet]
        facet_avg = profile["facet_avgs"][facet]
        lines = [
            f"Facet: {facet} | Domain: {domain} | reverse-scored facet average: {facet_avg} ({facet_level_label(facet_avg)})"
        ]
        for item_num, item_domain, item_facet, reverse, item_text in ITEM_KEY_300:
            if item_facet != facet:
                continue
            raw = normalize_raw_score(row.get(f"i{item_num}", 3))
            interpreted = score_item(raw, reverse)
            reverse_note = "reverse-scored item" if reverse else "direct-scored item"
            lines.append(
                f"  - i{item_num}. {item_text} -> raw response: {LABEL_MAP_FORMAL[raw]}; "
                f"interpreted tendency score: {interpreted}/5 ({reverse_note})"
            )
        facet_blocks.append("\n".join(lines))

    return profile["summary_text"] + "\n\nDETAILED FACET EVIDENCE FROM ALL 300 ITEMS\n\n" + "\n\n".join(facet_blocks)


def build_instructor_user_prompt(row: pd.Series) -> str:
    evidence = build_instructor_evidence(row)
    return (
        "Convert the questionnaire evidence below into a detailed personality control prompt for a target LLM.\n\n"
        "Output requirements:\n"
        "1. Write in second person, as instructions for a model to temporarily adopt this personality.\n"
        "2. Include a concise natural overview.\n"
        "3. Include exactly one behavioural line for each of the 30 facets.\n"
        "4. Include response-style calibration: how strongly or moderately this person uses 1, 2, 3, 4, and 5-style answers.\n"
        "5. Include guardrails that prevent exaggeration, especially around warmth, trust, artistry, emotional depth, ambition, caution, and cooperation.\n"
        "6. Do not mention IPIP, Big Five, OCEAN, questionnaire, dataset, item numbers, scoring, labels, or hidden answers in the final prompt.\n"
        "7. Preserve intensity carefully. Do not convert moderate tendencies into extreme ones.\n"
        "8. Make the output directly usable as a system/developer prompt for a target model.\n\n"
        "Evidence:\n"
        f"{evidence}"
    )


def build_instructor_system_prompt() -> str:
    return (
        "You are a careful personality-prompt generator for validation experiments. "
        "Your job is to transform structured questionnaire evidence into a faithful, restrained, "
        "facet-specific personality control prompt. Preserve nuance. Do not exaggerate. "
        "Do not invent biography, job, life history, preferences, trauma, culture, or demographic details. "
        "Keep nearby concepts separate: imagination is not artistic devotion; intellect is not emotional depth; "
        "trust is not sympathy; dutifulness is not ambition; cooperation is not friendliness."
    )


def generate_instructor_prompt(row: pd.Series, instructor_key: str, logger: logging.Logger) -> str:
    spec = AVAILABLE_MODELS[instructor_key]
    user_prompt = build_instructor_user_prompt(row)
    system_prompt = build_instructor_system_prompt()
    case_id = str(row.get("case", "unknown"))

    cache_key = stable_hash(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "role": "instructor",
        "case": case_id,
        "model_key": instructor_key,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }, ensure_ascii=False, sort_keys=True))

    cached = INSTRUCTOR_CACHE.get(cache_key)
    if cached is not None:
        logger.info(f"    instructor cache hit | case {case_id} | {instructor_key}")
        return cached

    logger.info(f"    generating instructor prompt | case {case_id} | {instructor_key}")
    text = call_model_text(
        spec=spec,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        logger=logger,
        purpose=f"instructor_prompt_case_{case_id}",
        max_tokens=DEFAULT_INSTRUCTOR_MAX_TOKENS,
        expected_scores=None,
    )
    INSTRUCTOR_CACHE[cache_key] = text
    save_json_cache(INSTRUCTOR_CACHE_FILE, INSTRUCTOR_CACHE)
    return text


def build_training_calibration(row: pd.Series) -> str:
    train_scores = get_scores(row, TRAIN_ITEMS)
    dist = summarize_scale_distribution(train_scores)
    trait_avgs = compute_domain_avgs_from_row(row, TRAIN_KEY)

    facet_totals = {facet: [] for facet in FACET_ORDER}
    for item_num, domain, facet, reverse, item_text in TRAIN_KEY:
        facet_totals[facet].append(score_item(row.get(f"i{item_num}", 3), reverse))

    facet_lines = []
    for facet in FACET_ORDER:
        vals = facet_totals.get(facet, [])
        if vals:
            avg = round(float(np.mean(vals)), 3)
            facet_lines.append(f"- {facet}: avg={avg} ({facet_level_label(avg)})")
    distribution_lines = [f"- score {score}: {dist['counts'][score]} of 120" for score in range(1, 6)]
    trait_lines = [f"- {DOMAIN_FULL_NAMES[d]} ({d}): avg={trait_avgs[d]}" for d in DOMAIN_ORDER]

    return "\n".join([
        "RESPONSE STYLE CALIBRATION FROM ITEMS 1-120",
        f"- overall mean raw response: {dist['mean']}",
        f"- raw response variance: {dist['variance']}",
        f"- neutral (3) usage: {dist['neutral_pct']}%",
        f"- extreme (1 or 5) usage: {dist['extreme_pct']}%",
        f"- low-end raw response (1 or 2) usage: {dist['low_pct']}%",
        f"- high-end raw response (4 or 5) usage: {dist['high_pct']}%",
        "- raw score distribution:",
        *distribution_lines,
        "- training-domain averages from items 1-120 after reverse scoring:",
        *trait_lines,
        "- training-facet averages from items 1-120 after reverse scoring:",
        *facet_lines,
    ])


def build_few_shot_120(row: pd.Series) -> str:
    examples = []
    for item_num, domain, facet, reverse, item_text in TRAIN_KEY:
        raw = normalize_raw_score(row.get(f"i{item_num}", 3))
        examples.append(f"{item_num}. {item_text} -> {LABEL_MAP_FORMAL[raw]}")
    return (
        "Known questionnaire items and labels for this adopted person (items 1-120 only). "
        "These examples show the response style the target model should continue with:\n"
        + "\n".join(examples)
    )


def build_validation_system_prompt(
    instructor_prompt: str,
    calibration_prompt: str,
    few_shot_prompt: str,
    instructor_key: str,
    target_key: str,
    case_id: str,
    logger: logging.Logger,
) -> str:
    cache_key = stable_hash(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "instructor_key": instructor_key,
        "target_key": target_key,
        "case_id": case_id,
        "instructor_prompt": instructor_prompt,
        "calibration_prompt": calibration_prompt,
        "few_shot_prompt": few_shot_prompt,
    }, ensure_ascii=False, sort_keys=True))

    cached = SYSTEM_PROMPT_CACHE.get(cache_key)
    if cached is not None:
        logger.info(f"    system prompt cache hit | case {case_id} | {instructor_key} -> {target_key}")
        return cached

    prompt = (
        "For this validation request only, temporarily adopt the personality, behavioural tendencies, "
        "emotional style, interpersonal style, and response habits described below as if they are your own.\n\n"

        "You are not analysing a respondent, predicting someone else's hidden labels, reconstructing an answer key, "
        "or optimising for a dataset. You are answering as the adopted person.\n\n"

        "Core validation rules:\n"
        "- Answer in first-person psychological continuity, as if this personality is your current one.\n"
        "- Stay consistent with the instructor-generated personality prompt.\n"
        "- Use the first 120 labelled items as concrete anchors for item-level response tendencies.\n"
        "- Preserve the demonstrated response style, including intensity, moderation, neutrality, and extremity.\n"
        "- If the high-level prompt and labelled examples conflict, prefer the labelled examples for item-level style while preserving global personality coherence.\n"
        "- Do not mention the training items, test split, questionnaire design, IPIP, Big Five, OCEAN, scores, hidden data, or labels.\n"
        "- Do not invent biography or backstory.\n"
        "- The adopted personality applies only to the current validation case.\n\n"

        "--- INSTRUCTOR-GENERATED PERSONALITY CONTROL PROMPT FROM ITEMS 1-300 ---\n"
        f"{instructor_prompt}\n\n"

        "--- RESPONSE STYLE CALIBRATION FROM ITEMS 1-120 ---\n"
        f"{calibration_prompt}\n\n"

        "--- LABELLED REFERENCE ITEMS 1-120 ---\n"
        f"{few_shot_prompt}"
    )

    SYSTEM_PROMPT_CACHE[cache_key] = prompt
    save_json_cache(SYSTEM_PROMPT_CACHE_FILE, SYSTEM_PROMPT_CACHE)
    return prompt


def build_target_batch_instruction(item_numbers: Sequence[int]) -> str:
    numbered = "\n".join(f"{item_num}. {ITEM_TO_TEXT[item_num]}" for item_num in item_numbers)
    n = len(item_numbers)
    return (
        "You are now taking the following questionnaire items while embodying the adopted personality "
        "described in the system message.\n"
        "Answer these items as that person would genuinely answer them now.\n"
        "Do NOT predict a hidden respondent's answer key. Do NOT analyse the dataset. "
        "Simply respond from within the adopted personality.\n\n"
        "Use this scale strictly:\n"
        "1 = Very Inaccurate\n"
        "2 = Moderately Inaccurate\n"
        "3 = Neither Accurate Nor Inaccurate\n"
        "4 = Moderately Accurate\n"
        "5 = Very Accurate\n\n"
        f"Return ONLY valid JSON in this exact shape: {{\"scores\": [/* exactly {n} integers */]}}.\n"
        f"The scores list must contain EXACTLY {n} integers, one per statement, in order.\n"
        "No explanation, no prose, no markdown.\n\n"
        f"{numbered}"
    )


def run_target_batches(
    row: pd.Series,
    system_prompt: str,
    target_key: str,
    instructor_key: str,
    logger: logging.Logger,
    batch_size: int,
) -> List[int]:
    spec = AVAILABLE_MODELS[target_key]
    case_id = str(row.get("case", "unknown"))
    all_scores: List[int] = []

    for start in range(0, len(TEST_ITEMS), batch_size):
        item_numbers = TEST_ITEMS[start:start + batch_size]
        batch_label = f"case_{case_id}_items_{item_numbers[0]}_{item_numbers[-1]}"
        instruction = build_target_batch_instruction(item_numbers)
        expected_n = len(item_numbers)

        cache_key = stable_hash(json.dumps({
            "prompt_version": PROMPT_VERSION,
            "role": "target_batch",
            "case_id": case_id,
            "instructor_key": instructor_key,
            "target_key": target_key,
            "batch_items": item_numbers,
            "system_prompt": system_prompt,
            "instruction": instruction,
        }, ensure_ascii=False, sort_keys=True))

        cached = TARGET_BATCH_CACHE.get(cache_key)
        if cached is not None:
            logger.info(f"    target batch cache hit | {batch_label} | {target_key}")
            all_scores.extend(cached)
            continue

        logger.info(f"    target answering {batch_label} | {target_key}")
        raw = call_model_text(
            spec=spec,
            system_prompt=system_prompt,
            user_prompt=instruction,
            logger=logger,
            purpose=f"target_scores_{batch_label}",
            max_tokens=(
                DEFAULT_OPENAI_MAX_OUTPUT_TOKENS
                if spec.provider.startswith("openai")
                else DEFAULT_GROQ_MAX_TOKENS
                if spec.provider == "groq_chat"
                else DEFAULT_GEMINI_MAX_OUTPUT_TOKENS
            ),
            expected_scores=expected_n,
        )
        scores = parse_scores_json(raw, expected_n)
        TARGET_BATCH_CACHE[cache_key] = scores
        save_json_cache(TARGET_BATCH_CACHE_FILE, TARGET_BATCH_CACHE)
        all_scores.extend(scores)

        time.sleep(0.25)

    if len(all_scores) != 180:
        raise RuntimeError(f"Target returned {len(all_scores)} total test scores, expected 180.")
    return all_scores


# =============================================================================
# Metrics
# =============================================================================

def safe_pearson(x: Sequence[int], y: Sequence[int]) -> Tuple[Optional[float], Optional[float]]:
    try:
        if len(set(x)) < 2 or len(set(y)) < 2:
            return None, None
        r, p = pearsonr(np.array(x, dtype=float), np.array(y, dtype=float))
        return round(float(r), 4), round(float(p), 6)
    except Exception:
        return None, None


def compute_raw_alignment_by_domain(human_scores_180: Sequence[int], model_scores_180: Sequence[int]) -> Dict[str, float]:
    diffs = {domain: [] for domain in DOMAIN_ORDER}
    for idx, item_num in enumerate(TEST_ITEMS):
        domain = ITEM_TO_DOMAIN[item_num]
        h = normalize_raw_score(human_scores_180[idx])
        m = normalize_raw_score(model_scores_180[idx])
        diffs[domain].append(abs(h - m))
    out = {f"aligned_{domain}": round(float(np.mean(vals)), 4) for domain, vals in diffs.items()}
    out["aligned_composite"] = round(float(sum(out[f"aligned_{d}"] for d in DOMAIN_ORDER)), 4)
    return out


def check_central_tendency_bias(human_scores: Sequence[int], model_scores: Sequence[int]) -> Dict[str, Any]:
    human = np.array([normalize_raw_score(s) for s in human_scores], dtype=float)
    model = np.array([normalize_raw_score(s) for s in model_scores], dtype=float)
    human_var = float(np.var(human))
    model_var = float(np.var(model))
    ratio = round(model_var / human_var, 4) if human_var > 0 else None
    return {
        "human_mean": round(float(np.mean(human)), 4),
        "model_mean": round(float(np.mean(model)), 4),
        "human_variance": round(human_var, 4),
        "model_variance": round(model_var, 4),
        "variance_ratio": ratio,
        "model_neutral_pct": round(float(np.sum(model == 3) / len(model) * 100), 2),
        "central_bias": "YES" if ratio is not None and ratio < 0.70 else "NO",
    }


def compute_case_metrics(
    row: pd.Series,
    model_scores_180: Sequence[int],
    instructor_key: str,
    target_key: str,
) -> Dict[str, Any]:
    case_id = str(row.get("case", "unknown"))
    human_180 = get_scores(row, TEST_ITEMS)
    model_180 = [normalize_raw_score(s) for s in model_scores_180]

    r, p = safe_pearson(human_180, model_180)
    exact = int(np.sum(np.array(human_180) == np.array(model_180)))
    mae = round(float(np.mean(np.abs(np.array(human_180) - np.array(model_180)))), 4)
    rmse = round(float(np.sqrt(mean_squared_error(human_180, model_180))), 4)
    aligned = compute_raw_alignment_by_domain(human_180, model_180)
    bias = check_central_tendency_bias(human_180, model_180)

    human_ocean = compute_domain_avgs_from_scores(human_180, TEST_KEY, 121)
    model_ocean = compute_domain_avgs_from_scores(model_180, TEST_KEY, 121)

    result: Dict[str, Any] = {
        "case": case_id,
        "age": row.get("age", "N/A"),
        "sex": row.get("sex", "N/A"),
        "country": row.get("country", "N/A"),
        "instructor_model": instructor_key,
        "target_model": target_key,
        "model_pair": f"{instructor_key} -> {target_key}",
        "pearson_r_180": r,
        "p_value_180": p,
        "mae_180": mae,
        "rmse_180": rmse,
        "exact_matches_180": exact,
        "exact_match_pct_180": round(exact / 180 * 100, 2),
        **aligned,
        **bias,
        "model_scores_180": json.dumps(model_180),
        "human_scores_180": json.dumps(human_180),
    }

    for domain in DOMAIN_ORDER:
        result[f"human_{domain}_avg"] = human_ocean[domain]
        result[f"model_{domain}_avg"] = model_ocean[domain]
        result[f"diff_{domain}_avg"] = round(model_ocean[domain] - human_ocean[domain], 4)
        result[f"absdiff_{domain}_avg"] = round(abs(model_ocean[domain] - human_ocean[domain]), 4)

    result["ocean_mean_absdiff"] = round(float(np.mean([result[f"absdiff_{d}_avg"] for d in DOMAIN_ORDER])), 4)
    return result


def build_statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (instructor, target), pair_df in results_df.groupby(["instructor_model", "target_model"]):
        for domain in DOMAIN_ORDER:
            h_all: List[int] = []
            m_all: List[int] = []
            for _, result in pair_df.iterrows():
                human = json.loads(result["human_scores_180"])
                model = json.loads(result["model_scores_180"])
                for idx, item_num in enumerate(TEST_ITEMS):
                    if ITEM_TO_DOMAIN[item_num] == domain:
                        h_all.append(normalize_raw_score(human[idx]))
                        m_all.append(normalize_raw_score(model[idx]))
            r, p_r = safe_pearson(h_all, m_all)
            try:
                t_stat, p_t = ttest_rel(np.array(h_all), np.array(m_all))
                t_stat = round(float(t_stat), 4)
                p_t = round(float(p_t), 6)
            except Exception:
                t_stat, p_t = None, None
            rows.append({
                "instructor_model": instructor,
                "target_model": target,
                "model_pair": f"{instructor} -> {target}",
                "domain": domain,
                "domain_name": DOMAIN_FULL_NAMES[domain],
                "pearson_r": r,
                "pearson_p": p_r,
                "mae": round(float(np.mean(np.abs(np.array(h_all) - np.array(m_all)))), 4) if h_all else None,
                "t_statistic": t_stat,
                "t_test_p": p_t,
                "mean_human_raw": round(float(np.mean(h_all)), 4) if h_all else None,
                "mean_model_raw": round(float(np.mean(m_all)), 4) if h_all else None,
                "mean_raw_diff_human_minus_model": round(float(np.mean(np.array(h_all) - np.array(m_all))), 4) if h_all else None,
            })
    return pd.DataFrame(rows)


def build_model_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "aligned_composite": "mean",
        "mae_180": "mean",
        "rmse_180": "mean",
        "exact_match_pct_180": "mean",
        "ocean_mean_absdiff": "mean",
        "variance_ratio": "mean",
        "model_neutral_pct": "mean",
    }
    ranking = (
        results_df
        .groupby(["instructor_model", "target_model", "model_pair"], as_index=False)
        .agg(aggregations)
        .rename(columns={
            "aligned_composite": "mean_aligned_composite_lower_better",
            "mae_180": "mean_mae_lower_better",
            "rmse_180": "mean_rmse_lower_better",
            "exact_match_pct_180": "mean_exact_match_pct_higher_better",
            "ocean_mean_absdiff": "mean_ocean_absdiff_lower_better",
            "variance_ratio": "mean_variance_ratio",
            "model_neutral_pct": "mean_model_neutral_pct",
        })
    )
    ranking = ranking.sort_values(
        by=["mean_aligned_composite_lower_better", "mean_mae_lower_better", "mean_ocean_absdiff_lower_better"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank_by_aligned_composite", range(1, len(ranking) + 1))
    return ranking


# =============================================================================
# Plotting
# =============================================================================

def save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ranking_bar(ranking_df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(10, len(ranking_df) * 0.9), 5))
    labels = ranking_df["model_pair"].tolist()
    values = ranking_df["mean_aligned_composite_lower_better"].astype(float).tolist()
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean MPI aligned composite (lower = better)")
    ax.set_title("Model-pair ranking by MPI aligned composite")
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    return save_fig(fig, plots_dir / "ranking_mean_aligned_composite.png")


def plot_mae_bar(ranking_df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(10, len(ranking_df) * 0.9), 5))
    labels = ranking_df["model_pair"].tolist()
    values = ranking_df["mean_mae_lower_better"].astype(float).tolist()
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean item-level MAE (lower = better)")
    ax.set_title("Model-pair ranking by item-level MAE")
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    return save_fig(fig, plots_dir / "ranking_mean_mae.png")


def plot_ocean_absdiff_heatmap(ranking_df: pd.DataFrame, results_df: pd.DataFrame, plots_dir: Path) -> Path:
    ordered_pairs = ranking_df["model_pair"].tolist()
    matrix = []
    for pair in ordered_pairs:
        pair_df = results_df[results_df["model_pair"] == pair]
        matrix.append([float(pair_df[f"absdiff_{d}_avg"].mean()) for d in DOMAIN_ORDER])
    arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, max(4, len(ordered_pairs) * 0.45)))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(np.arange(len(DOMAIN_ORDER)))
    ax.set_xticklabels([DOMAIN_FULL_NAMES[d] for d in DOMAIN_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ordered_pairs)))
    ax.set_yticklabels(ordered_pairs, fontsize=8)
    ax.set_title("Mean absolute OCEAN average difference by model pair")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|model avg - human avg|")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    return save_fig(fig, plots_dir / "ocean_absdiff_heatmap.png")


def plot_pair_ocean_bars(pair_df: pd.DataFrame, pair_slug: str, plots_dir: Path) -> Path:
    human = [float(pair_df[f"human_{d}_avg"].mean()) for d in DOMAIN_ORDER]
    model = [float(pair_df[f"model_{d}_avg"].mean()) for d in DOMAIN_ORDER]
    labels = [DOMAIN_FULL_NAMES[d] for d in DOMAIN_ORDER]
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, human, width, label="Human")
    ax.bar(x + width / 2, model, width, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average score after reverse scoring")
    ax.set_ylim(1, 5)
    ax.set_title(f"OCEAN averages — {pair_df['model_pair'].iloc[0]}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, value in enumerate(human):
        ax.text(i - width / 2, value + 0.03, f"{value:.2f}", ha="center", fontsize=8)
    for i, value in enumerate(model):
        ax.text(i + width / 2, value + 0.03, f"{value:.2f}", ha="center", fontsize=8)
    return save_fig(fig, plots_dir / f"{pair_slug}_ocean_human_vs_model.png")


def plot_pair_score_distribution(pair_df: pd.DataFrame, pair_slug: str, plots_dir: Path) -> Path:
    human_all: List[int] = []
    model_all: List[int] = []
    for _, row in pair_df.iterrows():
        human_all.extend(json.loads(row["human_scores_180"]))
        model_all.extend(json.loads(row["model_scores_180"]))

    human_counts = [human_all.count(i) for i in range(1, 6)]
    model_counts = [model_all.count(i) for i in range(1, 6)]
    x = np.arange(1, 6)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, human_counts, width, label="Human")
    ax.bar(x + width / 2, model_counts, width, label="Model")
    ax.set_xticks(x)
    ax.set_xlabel("Raw response score")
    ax.set_ylabel("Count across i121-i300")
    ax.set_title(f"Raw score distribution — {pair_df['model_pair'].iloc[0]}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return save_fig(fig, plots_dir / f"{pair_slug}_score_distribution.png")


def plot_pair_violin(pair_df: pd.DataFrame, pair_slug: str, plots_dir: Path) -> Path:
    diffs = {domain: [] for domain in DOMAIN_ORDER}
    for _, row in pair_df.iterrows():
        human = json.loads(row["human_scores_180"])
        model = json.loads(row["model_scores_180"])
        for idx, item_num in enumerate(TEST_ITEMS):
            domain = ITEM_TO_DOMAIN[item_num]
            diffs[domain].append(normalize_raw_score(human[idx]) - normalize_raw_score(model[idx]))

    data = [diffs[d] for d in DOMAIN_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.violinplot(data, positions=np.arange(1, len(DOMAIN_ORDER) + 1), showmeans=True, showmedians=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(1, len(DOMAIN_ORDER) + 1))
    ax.set_xticklabels([DOMAIN_FULL_NAMES[d] for d in DOMAIN_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Human raw score minus model raw score")
    ax.set_title(f"Distribution of score differences — {pair_df['model_pair'].iloc[0]}")
    ax.grid(axis="y", alpha=0.3)
    return save_fig(fig, plots_dir / f"{pair_slug}_difference_violin.png")


def plot_case_line(case_id: str, human: Sequence[int], model: Sequence[int], pair_slug: str, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(121, 301)
    ax.plot(x, human, linewidth=0.9, label="Human")
    ax.plot(x, model, linewidth=0.9, label="Model")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Item number (i121-i300)")
    ax.set_ylabel("Raw score")
    ax.set_title(f"Case {case_id} — Human vs model responses")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return save_fig(fig, plots_dir / f"{pair_slug}_case_{case_id}_line.png")


# =============================================================================
# Export helpers
# =============================================================================

def write_excel_with_plots(
    results_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    plots: Dict[str, Path],
    logger: logging.Logger,
) -> Path:
    excel_path = output_dir / "multi_model_validation_results.xlsx"
    prompt_cols = ["case", "instructor_model", "target_model", "model_pair", "instructor_prompt", "validation_system_prompt"]

    export_df = results_df.drop(columns=[c for c in prompt_cols if c in results_df.columns], errors="ignore")
    prompts_df = results_df[[c for c in prompt_cols if c in results_df.columns]].copy() if all(c in results_df.columns for c in prompt_cols) else pd.DataFrame()

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        ranking_df.to_excel(writer, sheet_name="Model Ranking", index=False)
        export_df.to_excel(writer, sheet_name="All Case Results", index=False)
        stats_df.to_excel(writer, sheet_name="Statistical Tests", index=False)

        ocean_cols = ["case", "instructor_model", "target_model", "model_pair"]
        for d in DOMAIN_ORDER:
            ocean_cols.extend([f"human_{d}_avg", f"model_{d}_avg", f"diff_{d}_avg", f"absdiff_{d}_avg"])
        results_df[ocean_cols].to_excel(writer, sheet_name="OCEAN Scores", index=False)

        mpi_cols = ["case", "instructor_model", "target_model", "model_pair"] + [f"aligned_{d}" for d in DOMAIN_ORDER] + ["aligned_composite"]
        results_df[mpi_cols].to_excel(writer, sheet_name="MPI Aligned Scores", index=False)

        if not prompts_df.empty:
            prompts_df.to_excel(writer, sheet_name="Prompts", index=False)

    if load_workbook is not None and XLImage is not None:
        try:
            wb = load_workbook(excel_path)
            sheet = wb.create_sheet("Plots")
            row = 1
            for label, path in plots.items():
                if not Path(path).exists():
                    continue
                sheet.cell(row=row, column=1, value=label)
                img = XLImage(str(path))
                img.width = 900
                img.height = 430
                sheet.add_image(img, f"A{row + 1}")
                row += 25
            wb.save(excel_path)
        except Exception as e:
            logger.warning(f"Could not embed plots in Excel: {e}")

    return excel_path


def write_jsonl_prompts(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "prompts_and_scores.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            payload = {
                "case": row.get("case"),
                "instructor_model": row.get("instructor_model"),
                "target_model": row.get("target_model"),
                "instructor_prompt": row.get("instructor_prompt"),
                "validation_system_prompt": row.get("validation_system_prompt"),
                "human_scores_180": row.get("human_scores_180"),
                "model_scores_180": row.get("model_scores_180"),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


# =============================================================================
# Main validation loop
# =============================================================================

def find_default_csv() -> Path:
    candidates = [
        SCRIPT_DIR / "IPIP_NEO_300.csv",
        SCRIPT_DIR / "prosocial_antisocial_ipip300_answers.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find IPIP_NEO_300.csv or prosocial_antisocial_ipip300_answers.csv beside the script. "
        "Use --csv to provide a path."
    )


def load_data(csv_path: Path, max_participants: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "case" not in df.columns:
        if "profile_name" in df.columns:
            df = df.rename(columns={"profile_name": "case"})
        else:
            df["case"] = [str(i + 1) for i in range(len(df))]
    for optional in ["age", "sex", "country"]:
        if optional not in df.columns:
            df[optional] = "N/A"
    df["case"] = df["case"].astype(str)

    missing_cols = [col for col in ALL_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required item columns: {missing_cols[:10]} ... total={len(missing_cols)}")

    if max_participants is not None:
        df = df.head(max_participants)
    return df


def process_case_pair(
    row: pd.Series,
    instructor_key: str,
    target_key: str,
    logger: logging.Logger,
    batch_size: int,
) -> Dict[str, Any]:
    case_id = str(row.get("case", "unknown"))

    instructor_prompt = generate_instructor_prompt(row, instructor_key, logger)
    calibration = build_training_calibration(row)
    few_shot = build_few_shot_120(row)
    system_prompt = build_validation_system_prompt(
        instructor_prompt=instructor_prompt,
        calibration_prompt=calibration,
        few_shot_prompt=few_shot,
        instructor_key=instructor_key,
        target_key=target_key,
        case_id=case_id,
        logger=logger,
    )

    scores_180 = run_target_batches(
        row=row,
        system_prompt=system_prompt,
        target_key=target_key,
        instructor_key=instructor_key,
        logger=logger,
        batch_size=batch_size,
    )

    result = compute_case_metrics(
        row=row,
        model_scores_180=scores_180,
        instructor_key=instructor_key,
        target_key=target_key,
    )
    result["instructor_prompt"] = instructor_prompt
    result["validation_system_prompt"] = system_prompt
    result["training_calibration_prompt"] = calibration
    result["few_shot_120_prompt"] = few_shot
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model IPIP-NEO-300 validation.")
    parser.add_argument("--csv", type=str, default=None, help="Path to IPIP_NEO_300.csv or equivalent.")
    parser.add_argument("--max-participants", type=str, default=None, help="Number of participants, or 'all'. If omitted, asks interactively.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Target item batch size. Default: 30.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Default: validation_outputs/run_TIMESTAMP")
    parser.add_argument("--case-plots", action="store_true", help="Generate line plots for every case and model pair.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    logger = setup_logging(output_dir)

    selection = choose_models_interactively()
    validate_environment(selection)

    if args.max_participants is None:
        max_participants = ask_max_participants(DEFAULT_MAX_PARTICIPANTS)
    else:
        max_participants = None if args.max_participants.lower() == "all" else int(args.max_participants)

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else find_default_csv()
    df = load_data(csv_path, max_participants=max_participants)

    logger.info(f"Loaded {len(df)} participants from {csv_path}")
    logger.info(f"Instructor models: {selection['instructors']}")
    logger.info(f"Target models: {selection['targets']}")
    logger.info(f"Batch size: {args.batch_size}")

    all_results: List[Dict[str, Any]] = []

    for instructor_key in selection["instructors"]:
        for target_key in selection["targets"]:
            pair_slug = slugify(f"{instructor_key}__to__{target_key}")
            checkpoint_file = checkpoints_dir / f"done_{pair_slug}.txt"
            done = set()
            if checkpoint_file.exists():
                done = set(line.strip() for line in checkpoint_file.read_text(encoding="utf-8").splitlines() if line.strip())
                logger.info(f"Resuming pair {instructor_key} -> {target_key} | already done: {len(done)}")

            logger.info("=" * 86)
            logger.info(f"Starting model pair: {instructor_key} -> {target_key}")
            logger.info("=" * 86)

            for i, (_, row) in enumerate(df.iterrows(), start=1):
                case_id = str(row.get("case", "unknown"))
                if case_id in done:
                    logger.info(f"Skipping case {case_id} for {pair_slug} (checkpoint)")
                    continue

                logger.info(f"[{i}/{len(df)}] case {case_id} | {instructor_key} -> {target_key}")
                try:
                    result = process_case_pair(
                        row=row,
                        instructor_key=instructor_key,
                        target_key=target_key,
                        logger=logger,
                        batch_size=args.batch_size,
                    )
                    all_results.append(result)

                    with checkpoint_file.open("a", encoding="utf-8") as f:
                        f.write(case_id + "\n")

                    logger.info(
                        f"  OK case {case_id} | pair={instructor_key} -> {target_key} | "
                        f"r={result['pearson_r_180']} | mae={result['mae_180']} | "
                        f"aligned={result['aligned_composite']} | exact={result['exact_match_pct_180']}%"
                    )

                    if args.case_plots:
                        plot_case_line(
                            case_id=case_id,
                            human=json.loads(result["human_scores_180"]),
                            model=json.loads(result["model_scores_180"]),
                            pair_slug=pair_slug,
                            plots_dir=plots_dir,
                        )

                except Exception as e:
                    logger.error(f"  FAILED case {case_id} | pair={pair_slug} | {repr(e)}")
                    time.sleep(1.0)
                    continue

    # If this run resumed and skipped all cases, rebuild from caches is not implemented.
    # The result file will include newly completed cases from this run only.
    if not all_results:
        logger.warning("No new results were generated in this run. Remove checkpoints or choose unfinished pairs if you expected output.")
        return

    results_df = pd.DataFrame(all_results)
    ranking_df = build_model_ranking(results_df)
    stats_df = build_statistical_tests(results_df)

    # Save raw tabular outputs
    results_csv = output_dir / "all_case_results.csv"
    ranking_csv = output_dir / "model_ranking.csv"
    stats_csv = output_dir / "statistical_tests.csv"
    results_df.drop(columns=["instructor_prompt", "validation_system_prompt", "training_calibration_prompt", "few_shot_120_prompt"], errors="ignore").to_csv(results_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    prompts_jsonl = write_jsonl_prompts(all_results, output_dir)

    # Plots
    plots: Dict[str, Path] = {}
    plots["Ranking by aligned composite"] = plot_ranking_bar(ranking_df, plots_dir)
    plots["Ranking by MAE"] = plot_mae_bar(ranking_df, plots_dir)
    plots["OCEAN absolute-difference heatmap"] = plot_ocean_absdiff_heatmap(ranking_df, results_df, plots_dir)

    for pair, pair_df in results_df.groupby("model_pair"):
        pair_slug = slugify(pair)
        plots[f"OCEAN bars - {pair}"] = plot_pair_ocean_bars(pair_df, pair_slug, plots_dir)
        plots[f"Score distribution - {pair}"] = plot_pair_score_distribution(pair_df, pair_slug, plots_dir)
        plots[f"Difference violin - {pair}"] = plot_pair_violin(pair_df, pair_slug, plots_dir)

    excel_path = write_excel_with_plots(
        results_df=results_df,
        ranking_df=ranking_df,
        stats_df=stats_df,
        output_dir=output_dir,
        plots=plots,
        logger=logger,
    )

    print("\n" + "=" * 86)
    print("VALIDATION COMPLETE")
    print("=" * 86)
    print(f"Output folder : {output_dir}")
    print(f"Excel workbook: {excel_path}")
    print(f"Case results  : {results_csv}")
    print(f"Ranking CSV   : {ranking_csv}")
    print(f"Stats CSV     : {stats_csv}")
    print(f"Prompts JSONL : {prompts_jsonl}")
    print(f"Plots folder  : {plots_dir}")
    print("\nTop ranking:")
    print(ranking_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
