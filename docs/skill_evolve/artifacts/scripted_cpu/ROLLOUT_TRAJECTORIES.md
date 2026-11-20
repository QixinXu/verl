# Skill-evolution rollout trajectories

> Tokens produced by the SCRIPTED policy of examples/skill_evolve/scripted_alfworld_policy.py, not by a language model. This run validates the harness (data path, credit assignment, library), not model quality.

224 units (training and validation), 2184 episodes with advantages.

## Library after each step

| step | active skills | per task type | events |
|---|---|---|---|
| 1 | 3 | {"pick": 1, "look": 2} | admit sk0000; admit sk0001; admit sk0002; duplicate_dropped sk0000; duplicate_dropped sk0000; duplicate_dropped sk0001; duplicate_dropped sk0002; duplicate_dropped sk0000; duplicate_dropped sk0001; duplicate_dropped sk0002 |
| 2 | 3 | {"pick": 1, "look": 2} | merge_rejected ['sk0001', 'sk0002']; duplicate_dropped sk0000; duplicate_dropped sk0001; duplicate_dropped sk0002; duplicate_dropped sk0001; duplicate_dropped sk0000; duplicate_dropped sk0000; duplicate_replaced sk0000; duplicate_dropped sk0002; duplicate_dropped sk0001 |
| 3 | 3 | {"pick": 1, "look": 2} | retest sk0001; retest sk0002; duplicate_dropped sk0000; duplicate_replaced sk0002; duplicate_dropped sk0001; duplicate_dropped sk0002; duplicate_replaced sk0002; duplicate_dropped sk0001; duplicate_dropped sk0000 |
| 4 | 3 | {"pick": 1, "look": 2} | retest sk0000; duplicate_dropped sk0001; duplicate_dropped sk0002; duplicate_dropped sk0001; duplicate_dropped sk0000; duplicate_dropped sk0000; duplicate_replaced sk0000; duplicate_dropped sk0000; duplicate_replaced sk0002; duplicate_replaced sk0001; duplicate_replaced sk0002; duplicate_dropped sk0001 |
| 5 | 7 | {"pick": 1, "look": 2, "heat": 2, "clean": 2} | admit sk0003; admit sk0004; admit sk0005; admit sk0006; duplicate_dropped sk0004; duplicate_dropped sk0003; duplicate_dropped sk0006; duplicate_dropped sk0005 |
| 6 | 7 | {"pick": 1, "look": 2, "heat": 2, "clean": 2} | retest sk0001; retest sk0002; retest sk0000; duplicate_replaced sk0000; duplicate_dropped sk0000; duplicate_replaced sk0001; duplicate_replaced sk0002; duplicate_dropped sk0000; duplicate_dropped sk0002; duplicate_dropped sk0001 |
| 7 | 7 | {"pick": 1, "look": 2, "heat": 2, "clean": 2} | retest sk0005; retest sk0006; retest sk0004; retest sk0003; duplicate_dropped sk0005; duplicate_replaced sk0006; duplicate_replaced sk0006; duplicate_replaced sk0005; duplicate_replaced sk0003; duplicate_dropped sk0004 |
| 8 | 7 | {"pick": 1, "look": 2, "heat": 2, "clean": 2} | retest sk0002; retest sk0001; retest sk0000; duplicate_dropped sk0000; duplicate_dropped sk0000; duplicate_dropped sk0001; duplicate_dropped sk0001; duplicate_replaced sk0001; duplicate_dropped sk0002; duplicate_dropped sk0000 |

### Skills in the library after step 8

| id | status | task type | name | added | last verified | utility | loads | re-tests | strikes |
|---|---|---|---|---|---|---|---|---|---|
| sk0000 | active | pick | Find, take, place | 1 | 8 | +0.145 | 374 | 3 | 0 |
| sk0001 | active | look | Examine under a lamp | 1 | 8 | +0.158 | 169 | 3 | 0 |
| sk0002 | active | look | Examine under a lamp | 1 | 8 | +0.019 | 170 | 3 | 1 |
| sk0003 | active | heat | Heat with the microwave, then place | 5 | 7 | +0.002 | 31 | 1 | 0 |
| sk0004 | active | heat | Heat with the microwave, then place | 5 | 7 | +0.052 | 34 | 1 | 0 |
| sk0005 | active | clean | Clean at the sink basin, then place | 5 | 7 | +0.011 | 27 | 1 | 0 |
| sk0006 | active | clean | Clean at the sink basin, then place | 5 | 7 | +0.011 | 22 | 1 | 0 |

**sk0000 — Find, take, place** (from `train/pick_and_place_simple-Candle-None-Drawer-411/trial_T20190909_093542_770989`, step 1)

```text
When to use: The task is to put an object in or on a receptacle
1. Go to receptacles one at a time; open a closed one before looking inside.
2. When you see the object, take <object> from <receptacle>.
3. Go to the target receptacle.
4. Place it with: move <object> to <receptacle> (the game rejects 'put').
```

**sk0001 — Examine under a lamp** (from `train/look_at_obj_in_light-Book-None-DeskLamp-320/trial_T20190909_152445_177541`, step 1)

```text
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
```

**sk0002 — Examine under a lamp** (from `train/look_at_obj_in_light-Book-None-DeskLamp-320/trial_T20190909_152445_177541`, step 1)

```text
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
4. If the game rejects a command, read the commands it lists as available and use one of those.
```

**sk0003 — Heat with the microwave, then place** (from `train/pick_heat_then_place_in_recep-Tomato-None-SideTable-28/trial_T20190908_031531_364896`, step 5)

```text
When to use: The task is to heat an object and put it somewhere
1. Search receptacles until you see the object and take it.
2. Go to the microwave (the stove burners cannot heat).
3. Use: heat <object> with microwave.
4. Go to the target receptacle and use: move <object> to <receptacle>.
```

**sk0004 — Heat with the microwave, then place** (from `train/pick_heat_then_place_in_recep-Tomato-None-SideTable-28/trial_T20190908_031531_364896`, step 5)

```text
When to use: The task is to heat an object and put it somewhere
1. Search receptacles until you see the object and take it.
2. Go to the microwave (the stove burners cannot heat).
3. Use: heat <object> with microwave.
4. Go to the target receptacle and use: move <object> to <receptacle>.
5. If the game rejects a command, read the commands it lists as available and use one of those.
```

**sk0005 — Clean at the sink basin, then place** (from `train/pick_clean_then_place_in_recep-Spoon-None-DiningTable-20/trial_T20190909_034703_947964`, step 5)

```text
When to use: The task is to clean an object and put it somewhere
1. Search receptacles until you see the object and take it.
2. Go to the sinkbasin and use: clean <object> with sinkbasin.
3. Go to the target receptacle and use: move <object> to <receptacle>.
4. If the game rejects a command, read the commands it lists as available and use one of those.
```

**sk0006 — Clean at the sink basin, then place** (from `train/pick_clean_then_place_in_recep-Spoon-None-DiningTable-20/trial_T20190909_034703_947964`, step 5)

```text
When to use: The task is to clean an object and put it somewhere
1. Search receptacles until you see the object and take it.
2. Go to the sinkbasin and use: clean <object> with sinkbasin.
3. Go to the target receptacle and use: move <object> to <receptacle>.
```

---

## Unit 1 — a new skill is written, checked and credited (empty library)

Nothing can be loaded yet. Episodes on A write skills on their own initiative; every distinct skill is put into the index of the check tasks B1, B2, and its credit is what the episodes that really used it gained over the control episodes.

Step 1, task type `look`, library version 0 (uid `7693657f-585c-4a65-bda2-b0781c41172d`).

- A (source task): `look at book under the desklamp`
- B1 (check task): `look at pillow under the desklamp`, baseline score 0.846
- B2 (check task): `look at creditcard under the desklamp`, baseline score 0.675

### Arms

| arm | task | skill index in the prompt | episodes | mean score |
|---|---|---|---|---|
| `source` | look at book under the desklamp | (empty) | 8 | 0.823 |
| `check0/control` | look at pillow under the desklamp | (empty) | 4 | 0.846 |
| `check0/new0` | look at pillow under the desklamp | S1=new0 | 4 | 0.879 |
| `check0/new1` | look at pillow under the desklamp | S1=new1 | 4 | 0.667 |
| `check1/control` | look at creditcard under the desklamp | (empty) | 4 | 0.675 |
| `check1/new0` | look at creditcard under the desklamp | S1=new0 | 4 | 0.908 |
| `check1/new1` | look at creditcard under the desklamp | S1=new1 | 4 | 0.912 |

### New skills written on A and their credit

| candidate | name | written by (episode, call) | loaded | sum of usage weights | credit |
|---|---|---|---|---|---|
| new0 | Examine under a lamp | [[0, 0], [2, 0], [7, 0]] | 8/8 | 6.00 | +0.146 |
| new1 | Examine under a lamp | [[3, 0], [5, 0], [6, 0]] | 6/8 | 5.00 | +0.139 |

**new0 — Examine under a lamp**

```text
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
```

Credit of `new0`, episode by episode (credit = sum of u x (score - baseline) / (sum of u + kappa)):

| check task | episode | loaded | influence | u | score | baseline | u x (score - baseline) |
|---|---|---|---|---|---|---|---|
| B1 | 12 | yes | -0.517 | 0.00 | 0.867 | 0.846 | +0.000 |
| B1 | 13 | yes | +0.408 | 1.00 | 0.900 | 0.846 | +0.054 |
| B1 | 14 | yes | -0.517 | 0.00 | 0.867 | 0.846 | +0.000 |
| B1 | 15 | yes | +0.341 | 1.00 | 0.883 | 0.846 | +0.037 |
| B2 | 24 | yes | +2.000 | 1.00 | 0.900 | 0.675 | +0.225 |
| B2 | 25 | yes | +2.000 | 1.00 | 0.900 | 0.675 | +0.225 |
| B2 | 26 | yes | +0.488 | 1.00 | 0.917 | 0.675 | +0.242 |
| B2 | 27 | yes | +0.488 | 1.00 | 0.917 | 0.675 | +0.242 |

### Rewards of the source episodes

Ordinary tokens get the task advantage. The tokens of a write_skill call get the task advantage plus the credit of the skill, normalized over the group (an episode that wrote nothing counts as credit 0).

| episode | score of A | task advantage | wrote | credit | advantage of the write_skill tokens |
|---|---|---|---|---|---|
| 0 | 0.833 | +0.338 | new0@step9 | +0.146 | +0.935 |
| 1 | 0.850 | +0.880 | - | - | - |
| 2 | 0.750 | -2.369 | new0@step14 | +0.146 | -1.773 |
| 3 | 0.817 | -0.203 | new1@step10 | +0.139 | +0.279 |
| 4 | 0.833 | +0.338 | - | - | - |
| 5 | 0.833 | +0.338 | new1@step9 | +0.139 | +0.821 |
| 6 | 0.833 | +0.338 | new1@step9 | +0.139 | +0.821 |
| 7 | 0.833 | +0.338 | new0@step9 | +0.146 | +0.935 |

### Trajectories

**A source episode that wrote a skill** — arm `source`, episode 0: won in 10 steps, reward 0.833, advantage +0.338, advantage on its 83 write_skill tokens +0.935

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

- write_skill: saves what you learned in this task as a new skill for future tasks of the same kind. You may do this at any turn, also right after the task is completed. Reply like this:
Thought: <what is worth saving>
Action: write_skill
Name: <short title>
When to use: <one sentence describing the kind of task>
Procedure:
1. <step>
2. <step>
Write a general procedure: which steps in which order, which command works, how to recover from a typical mistake. It must not depend on this room, so never mention numbered objects such as "drawer 3". Future agents see only the name and the "when to use" line before they decide to load it. Use at most 120 words in the procedure.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a garbagecan 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.

Your task is to: examine the book with the desklamp.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 4, a book 3, a cellphone 1, a laptop 3, a laptop 2, a pillow 2, a pillow 1, and a teddybear 1.

[assistant]
Thought: I see book 4, take it.
Action: take book 4 from bed 1

[user]
You pick up the book 4 from the bed 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a creditcard 1, and a laptop 1.

[assistant]
Thought: I have not checked shelf 1 yet.
Action: go to shelf 1

[user]
You arrive at shelf 1. On the shelf 1, you see a keychain 1.

[assistant]
Thought: I have not checked desk 2 yet.
Action: go to desk 2

[user]
You arrive at desk 2. On the desk 2, you see a alarmclock 2, and a book 1.

[assistant]
Thought: I have not checked shelf 2 yet.
Action: go to shelf 2

[user]
You arrive at shelf 2. On the shelf 2, you see a alarmclock 1, a desklamp 1, and a pencil 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine book 4, examine shelf 2, inventory, look, move book 4 to shelf 2, use desklamp 1, go to bed 1, go to desk 1, go to desk 2, go to garbagecan 1, go to shelf 1, go to shelf 3, ... (+2 more)

[assistant]
Thought: Try the lamp.
Action: examine book 4 with desklamp 1

[user]
Nothing happens.
Commands available here: examine book 4, examine shelf 2, inventory, look, move book 4 to shelf 2, use desklamp 1, go to bed 1, go to desk 1, go to desk 2, go to garbagecan 1, go to shelf 1, go to shelf 3, ... (+2 more)

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1

[user]
You turn on the desklamp 1.
Task completed. You may save a skill with write_skill, or reply `Action: done`.

[assistant]
Thought: this is worth saving
Action: write_skill
Name: Examine under a lamp
When to use: The task is to look at an object under a lamp
Procedure:
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
```

**A check episode with the new skill new0 in its index** — arm `check0/new0`, episode 13: won in 6 steps, reward 0.900, advantage +1.306

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 3, a armchair 2, a armchair 1, a bed 1, a cabinet 2, a cabinet 1, a desk 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: look at pillow under the desklamp.

[assistant]
Thought: maybe this helps.
Action: load_skill S1

[user]
Loaded skill S1:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 1, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I see pillow 2, take it.
Action: take pillow 2 from bed 1

[user]
You pick up the pillow 2 from the bed 1.

[assistant]
Thought: Let me look somewhere else.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 3, a alarmclock 2, a cellphone 1, a houseplant 1, a mug 1, a pen 1, and a pencil 2.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a alarmclock 1, a desklamp 1, a keychain 1, and a pencil 1.

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```

**A check episode that did not use new0 (usage weight 0)** — arm `check0/new0`, episode 12: won in 8 steps, reward 0.867, advantage -0.783

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 3, a armchair 2, a armchair 1, a bed 1, a cabinet 2, a cabinet 1, a desk 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: look at pillow under the desklamp.

[assistant]
Thought: this skill is about my task.
Action: load_skill S1

[user]
Loaded skill S1:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 1, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I see pillow 2, take it.
Action: take pillow 2 from bed 1

[user]
You pick up the pillow 2 from the bed 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 3, a alarmclock 2, a cellphone 1, a houseplant 1, a mug 1, a pen 1, and a pencil 2.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a alarmclock 1, a desklamp 1, a keychain 1, and a pencil 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine pillow 2, examine sidetable 1, inventory, look, use desklamp 1, go to armchair 1, go to armchair 2, go to armchair 3, go to bed 1, go to cabinet 1, go to cabinet 2, go to desk 1, ... (+7 more)

[assistant]
Thought: Try the lamp.
Action: examine pillow 2 with desklamp 1

[user]
Nothing happens.
Commands available here: examine pillow 2, examine sidetable 1, inventory, look, use desklamp 1, go to armchair 1, go to armchair 2, go to armchair 3, go to bed 1, go to cabinet 1, go to cabinet 2, go to desk 1, ... (+7 more)

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```

**A control episode (old library only)** — arm `check0/control`, episode 8: won in 9 steps, reward 0.850, advantage +0.500

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 3, a armchair 2, a armchair 1, a bed 1, a cabinet 2, a cabinet 1, a desk 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: look at pillow under the desklamp.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 1, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I see pillow 2, take it.
Action: take pillow 2 from bed 1

[user]
You pick up the pillow 2 from the bed 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 3, a alarmclock 2, a cellphone 1, a houseplant 1, a mug 1, a pen 1, and a pencil 2.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a alarmclock 1, a desklamp 1, a keychain 1, and a pencil 1.

[assistant]
Thought: Let me look somewhere else.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 3, a alarmclock 2, a cellphone 1, a houseplant 1, a mug 1, a pen 1, and a pencil 2.

[assistant]
Thought: Go back to the lamp.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a alarmclock 1, a desklamp 1, a keychain 1, and a pencil 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine pillow 2, examine sidetable 1, inventory, look, use desklamp 1, go to armchair 1, go to armchair 2, go to armchair 3, go to bed 1, go to cabinet 1, go to cabinet 2, go to desk 1, ... (+7 more)

[assistant]
Thought: Try the lamp.
Action: examine pillow 2 with desklamp 1

[user]
Nothing happens.
Commands available here: examine pillow 2, examine sidetable 1, inventory, look, use desklamp 1, go to armchair 1, go to armchair 2, go to armchair 3, go to bed 1, go to cabinet 1, go to cabinet 2, go to desk 1, ... (+7 more)

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```


---

## Unit 2 — library skills are retrieved and loaded, and one is re-tested

The index of A now lists retrieved library skills; loading one is the policy's own action. The same unit re-tests a library skill of its task type with the same credit estimator: library with the skill against library without it.

Step 8, task type `pick`, library version 7 (uid `09cf3644-08df-4b57-a0e1-31ed637ad23c`).

- A (source task): `put a alarmclock in desk`
- B1 (check task): `put a keychain in armchair`, baseline score 0.833
- B2 (check task): `put a butterknife in sidetable`, baseline score 0.787

### Arms

| arm | task | skill index in the prompt | episodes | mean score |
|---|---|---|---|---|
| `source` | put a alarmclock in desk | S1=sk0004, S2=sk0005, S3=sk0006, S4=sk0003 | 8 | 0.858 |
| `check0/control` | put a keychain in armchair | S1=sk0004, S2=sk0000, S3=sk0005, S4=sk0003 | 4 | 0.833 |
| `check0/new0` | put a keychain in armchair | S1=sk0004, S2=sk0000, S3=sk0005, S4=sk0003, S5=new0 | 4 | 0.625 |
| `check0/new1` | put a keychain in armchair | S1=sk0004, S2=sk0000, S3=sk0005, S4=sk0003, S5=new1 | 4 | 0.625 |
| `check0/retest_sk0000/without` | put a keychain in armchair | S1=sk0004, S2=sk0005, S3=sk0003 | 4 | 0.600 |
| `check1/control` | put a butterknife in sidetable | S1=sk0004, S2=sk0000, S3=sk0005, S4=sk0003 | 4 | 0.787 |
| `check1/new0` | put a butterknife in sidetable | S1=new0, S2=sk0004, S3=sk0000, S4=sk0005, S5=sk0003 | 4 | 0.829 |
| `check1/new1` | put a butterknife in sidetable | S1=new1, S2=sk0004, S3=sk0000, S4=sk0005, S5=sk0003 | 4 | 0.404 |
| `check1/retest_sk0000/without` | put a butterknife in sidetable | S1=sk0004, S2=sk0005, S3=sk0003 | 4 | 0.792 |

### New skills written on A and their credit

| candidate | name | written by (episode, call) | loaded | sum of usage weights | credit |
|---|---|---|---|---|---|
| new0 | Find, take, place | [[0, 0], [1, 0], [2, 0], [4, 0], [7, 0]] | 4/8 | 4.00 | +0.028 |
| new1 | Be careful | [[3, 0], [5, 0]] | 1/8 | 0.00 | +0.000 |

**new0 — Find, take, place**

```text
When to use: The task is to put an object in or on a receptacle
1. Go to receptacles one at a time; open a closed one before looking inside.
2. When you see the object, take <object> from <receptacle>.
3. Go to the target receptacle.
4. Place it with: move <object> to <receptacle> (the game rejects 'put').
```

Credit of `new0`, episode by episode (credit = sum of u x (score - baseline) / (sum of u + kappa)):

| check task | episode | loaded | influence | u | score | baseline | u x (score - baseline) |
|---|---|---|---|---|---|---|---|
| B1 | 12 | yes | +0.226 | 1.00 | 0.833 | 0.833 | +0.000 |
| B1 | 13 | no | - | 0.00 | 0.000 | 0.833 | -0.000 |
| B1 | 14 | no | - | 0.00 | 0.833 | 0.833 | +0.000 |
| B1 | 15 | no | - | 0.00 | 0.833 | 0.833 | +0.000 |
| B2 | 28 | yes | +2.000 | 1.00 | 0.800 | 0.787 | +0.013 |
| B2 | 29 | no | - | 0.00 | 0.817 | 0.787 | +0.000 |
| B2 | 30 | yes | +0.230 | 1.00 | 0.850 | 0.787 | +0.062 |
| B2 | 31 | yes | +0.230 | 1.00 | 0.850 | 0.787 | +0.062 |

### Rewards of the source episodes

Ordinary tokens get the task advantage. The tokens of a write_skill call get the task advantage plus the credit of the skill, normalized over the group (an episode that wrote nothing counts as credit 0).

| episode | score of A | task advantage | wrote | credit | advantage of the write_skill tokens |
|---|---|---|---|---|---|
| 0 | 0.867 | +0.540 | new0@step7 | +0.028 | +1.265 |
| 1 | 0.867 | +0.540 | new0@step7 | +0.028 | +1.265 |
| 2 | 0.867 | +0.540 | new0@step7 | +0.028 | +1.265 |
| 3 | 0.833 | -1.620 | new1@step9 | +0.000 | -2.828 |
| 4 | 0.867 | +0.540 | new0@step7 | +0.028 | +1.265 |
| 5 | 0.867 | +0.540 | new1@step7 | +0.000 | -0.668 |
| 6 | 0.833 | -1.620 | - | - | - |
| 7 | 0.867 | +0.540 | new0@step7 | +0.028 | +1.265 |

**Re-test of library skill sk0000 (Find, take, place)**: loaded in 8/8 episodes, sum of usage weights 6.00, credit +0.145.

### Trajectories

**A source episode that wrote a skill** — arm `source`, episode 0: won in 8 steps, reward 0.867, advantage +0.540, advantage on its 114 write_skill tokens +1.265

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Heat with the microwave, then place - The task is to heat an object and put it somewhere
[S2] Clean at the sink basin, then place - The task is to clean an object and put it somewhere
[S3] Clean at the sink basin, then place - The task is to clean an object and put it somewhere
[S4] Heat with the microwave, then place - The task is to heat an object and put it somewhere
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

- write_skill: saves what you learned in this task as a new skill for future tasks of the same kind. You may do this at any turn, also right after the task is completed. Reply like this:
Thought: <what is worth saving>
Action: write_
... [549 more characters]

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, and a garbagecan 1.

Your task is to: put some alarmclock on desk.

[assistant]
Thought: maybe this helps.
Action: load_skill S4

[user]
Loaded skill S4:
### Heat with the microwave, then place
When to use: The task is to heat an object and put it somewhere
1. Search receptacles until you see the object and take it.
2. Go to the microwave (the stove burners cannot heat).
3. Use: heat <object> with microwave.
4. Go to the target receptacle and use: move <object> to <receptacle>.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a laptop 1, and a pillow 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a book 1, a cd 3, a cellphone 1, a desklamp 1, a keychain 3, and a keychain 2.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1, a cd 2, a creditcard 2, a creditcard 1, and a pencil 1.

[assistant]
Thought: I see alarmclock 1, take it.
Action: take alarmclock 1 from dresser 1

[user]
You pick up the alarmclock 1 from the dresser 1.

[assistant]
Thought: Carry it to desk 1.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a book 1, a cd 3, a cellphone 1, a desklamp 1, a keychain 3, and a keychain 2.

[assistant]
Thought: Place it with move.
Action: move alarmclock 1 to desk 1

[user]
You move the alarmclock 1 to the desk 1.
Task completed. You may save a skill with write_skill, or reply `Action: done`.

[assistant]
Thought: this is worth saving
Action: write_skill
Name: Find, take, place
When to use: The task is to put an object in or on a receptacle
Procedure:
1. Go to receptacles one at a time; open a closed one before looking inside.
2. When you see the object, take <object> from <receptacle>.
3. Go to the target receptacle.
4. Place it with: move <object> to <receptacle> (the game rejects 'put').
```

**A check episode with the new skill new0 in its index** — arm `check0/new0`, episode 12: won in 10 steps, reward 0.833, advantage +0.500

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Heat with the microwave, then place - The task is to heat an object and put it somewhere
[S2] Find, take, place - The task is to put an object in or on a receptacle
[S3] Clean at the sink basin, then place - The task is to clean an object and put it somewhere
[S4] Heat with the microwave, then place - The task is to heat an object and put it somewhere
[S5] Find, take, place - The task is to put an object in or on a receptacle
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 1, a bed 2, a bed 1, a cabinet 1, a desk 2, a desk 1, a drawer 13, a drawer 12, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 2, a dresser 1, and a garbagecan 1.

Your task is to: put a keychain in armchair.

[assistant]
Thought: this skill is about my task.
Action: load_skill S5

[user]
Loaded skill S5:
### Find, take, place
When to use: The task is to put an object in or on a receptacle
1. Go to receptacles one at a time; open a closed one before looking inside.
2. When you see the object, take <object> from <receptacle>.
3. Go to the target receptacle.
4. Place it with: move <object> to <receptacle> (the game rejects 'put').

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 2, and a pillow 2.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a desklamp 1, and a pencil 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1.

[assistant]
Thought: I have not checked bed 2 yet.
Action: go to bed 2

[user]
You arrive at bed 2. On the bed 2, you see a pillow 1.

[assistant]
Thought: I have not checked desk 2 yet.
Action: go to desk 2

[user]
You arrive at desk 2. On the desk 2, you see a alarmclock 2, a creditcard 2, a laptop 1, and a pen 1.

[assistant]
Thought: I have not checked dresser 2 yet.
Action: go to dresser 2

[user]
You arrive at dresser 2. On the dresser 2, you see a book 1, a keychain 1, and a mug 3.

[assistant]
Thought: I see keychain 1, take it.
Action: take keychain 1 from dresser 2

[user]
You pick up the keychain 1 from the dresser 2.

[assistant]
Thought: Carry it to armchair 1.
Action: go to armchair 1

[user]
You arrive at armchair 1. On the armchair 1, you see a cellphone 1.

[assistant]
Thought: Place it with move.
Action: move keychain 1 to armchair 1
```

**A check episode that did not use new0 (usage weight 0)** — arm `check0/new0`, episode 13: failed in 30 steps, reward 0.000, advantage -1.500

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Heat with the microwave, then place - The task is to heat an object and put it somewhere
[S2] Find, take, place - The task is to put an object in or on a receptacle
[S3] Clean at the sink basin, then place - The task is to clean an object and put it somewhere
[S4] Heat with the microwave, then place - The task is to heat an object and put it somewhere
[S5] Find, take, place - The task is to put an object in or on a receptacle
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 1, a bed 2, a bed 1, a cabinet 1, a desk 2, a desk 1, a drawer 13, a drawer 12, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 2, a dresser 1, and a garbagecan 1.

Your task is to: put a keychain in armchair.

[assistant]
Thought: Let me look somewhere else.
Action: go to garbagecan 1

[user]
You arrive at garbagecan 1. On the garbagecan 1, you see nothing.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 2, and a pillow 2.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a desklamp 1, and a pencil 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1.

[assistant]
Thought: I have not checked bed 2 yet.
Action: go to bed 2

[user]
You arrive at bed 2. On the bed 2, you see a pillow 1.

[assistant]
Thought: I have not checked desk 2 yet.
Action: go to desk 2

[user]
You arrive at desk 2. On the desk 2, you see a alarmclock 2, a creditcard 2, a laptop 1, and a pen 1.

[assistant]
Thought: I have not checked dresser 2 yet.
Action: go to dresser 2

[user]
You arrive at dresser 2. On the dresser 2, you see a book 1, a keychain 1, and a mug 3.

[assistant]
Thought: Let me look somewhere else.
Action: go to armchair 1

[user]
You arrive at armchair 1. On the armchair 1, you see a cellphone 1.

[assistant]
Thought: I have not checked drawer 1 yet.
Action: go to drawer 1

[user]
You arrive at drawer 1. The drawer 1 is closed.

[assistant]
Thought: drawer 1 is closed, open it.
Action: open drawer 1

[user]
You open the drawer 1. The drawer 1 is open. In it, you see nothing.

[assistant]
Thought: I have not checked drawer 2 yet.
Action: go to drawer 2

[user]
You arrive at drawer 2. The drawer 2 is closed.

[assistant]
Thought: drawer 2 is closed, open it.
Action: open drawer 2

[user]
You open the drawer 2. The drawer 2 is open. In it, you see nothing.

[assistant]
Thought: I have not checked drawer 3 yet.
Action: go to drawer 3

[user]
You arrive at drawer 3. The drawer 3 is closed.

[assistant]
Thought: drawer 3 is closed, open it.
Action: open drawer 3

[user]
You open the drawer 3. The drawer 3 is open. In it, you see a pencil 2.

[assistant]
Thought: Let me look somewhere else.
Action: go to drawer 8

[user]
You arrive at drawer 8. On the drawer 8, you see nothing.

[assistant]
Thought: I have not checked drawer 4 yet.
Action: go to drawer 4

[user]
You arrive at drawer 4. The drawer 4 is closed.

[assistant]
Thought: drawer 4 is closed, open it.
Action: open drawer 4

[user]
You open the drawer 4. The drawer 4 is open. In it, you see a creditcard 1.

[assistant]
Thought: I have not checked drawer 5 yet.
Action: go to drawer 5

[user]
You arrive at drawer 5. The drawer 5 is closed.

[assistant]
Thought: drawer 5 is closed, open it.
Action: open drawer 5

[user]
You open the drawer 5. The drawer 5 is open. In it, you see nothing.

[assistant]
Thought: I have not checked drawer 6 yet.
Action: go to drawer 6

[user]
You arrive at drawer 6. The drawer 6 is closed.

[assistant]
Thought: Let me look somewhere else.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1.

[assistant]
Thought: I have not checked drawer 7 yet.
Action: go to drawer 7

[user]
You arrive at drawer 7. On the drawer 7, you see nothing.

[assistant]
Thought: I have not checked drawer 9 yet.
Action: go to drawer 9

[user]
You arrive at drawer 9. On the drawer 9, you see nothing.

[assistant]
Thought: I have not checked drawer 10 yet.
Action: go to drawer 10

[user]
You arrive at drawer 10. The drawer 10 is closed.

[assistant]
Thought: drawer 10 is closed, open it.
Action: open drawer 10

[user]
You open the drawer 10. The drawer 10 is open. In it, you see nothing.

[assistant]
Thought: I have not checked drawer 11 yet.
Action: go to drawer 11

[user]
You arrive at drawer 11. The drawer 11 is closed.

[assistant]
Thought: drawer 11 is closed, open it.
Action: open drawer 11

[user]
You open the drawer 11. The drawer 11 is open. In it, you see nothing.

[assistant]
Thought: I have not checked drawer 12 yet.
Action: go to drawer 12

[user]
You arrive at drawer 12. The drawer 12 is closed.

[assistant]
Thought: drawer 12 is closed, open it.
Action: open drawer 12

[user]
You open the drawer 12. The drawer 12 is open. In it, you see a pencil 3.

[assistant]
Thought: I have not checked drawer 13 yet.
Action: go to drawer 13
```

**A control episode (old library only)** — arm `check0/control`, episode 8: won in 10 steps, reward 0.833, advantage +0.000

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Heat with the microwave, then place - The task is to heat an object and put it somewhere
[S2] Find, take, place - The task is to put an object in or on a receptacle
[S3] Clean at the sink basin, then place - The task is to clean an object and put it somewhere
[S4] Heat with the microwave, then place - The task is to heat an object and put it somewhere
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 1, a bed 2, a bed 1, a cabinet 1, a desk 2, a desk 1, a drawer 13, a drawer 12, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 2, a dresser 1, and a garbagecan 1.

Your task is to: put a keychain in armchair.

[assistant]
Thought: this skill is about my task.
Action: load_skill S2

[user]
Loaded skill S2:
### Find, take, place
When to use: The task is to put an object in or on a receptacle
1. Go to receptacles one at a time; open a closed one before looking inside.
2. When you see the object, take <object> from <receptacle>.
3. Go to the target receptacle.
4. Place it with: move <object> to <receptacle> (the game rejects 'put').

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 2, and a pillow 2.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a desklamp 1, and a pencil 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1.

[assistant]
Thought: I have not checked bed 2 yet.
Action: go to bed 2

[user]
You arrive at bed 2. On the bed 2, you see a pillow 1.

[assistant]
Thought: I have not checked desk 2 yet.
Action: go to desk 2

[user]
You arrive at desk 2. On the desk 2, you see a alarmclock 2, a creditcard 2, a laptop 1, and a pen 1.

[assistant]
Thought: I have not checked dresser 2 yet.
Action: go to dresser 2

[user]
You arrive at dresser 2. On the dresser 2, you see a book 1, a keychain 1, and a mug 3.

[assistant]
Thought: I see keychain 1, take it.
Action: take keychain 1 from dresser 2

[user]
You pick up the keychain 1 from the dresser 2.

[assistant]
Thought: Carry it to armchair 1.
Action: go to armchair 1

[user]
You arrive at armchair 1. On the armchair 1, you see a cellphone 1.

[assistant]
Thought: Place it with move.
Action: move keychain 1 to armchair 1
```


---

## Unit 3 — skills that earn no credit and are not admitted

No new skill of this unit has a positive credit: either the check episodes did not load it, or they loaded it without their actions depending on it, or using it did not beat the old library.

Step 1, task type `look`, library version 0 (uid `134b4e4f-6e82-4cdf-8a8b-7a33f10b8a10`).

- A (source task): `look at pen under the desklamp`
- B1 (check task): `look at alarmclock under the desklamp`, baseline score 0.854
- B2 (check task): `look at box under the desklamp`, baseline score 0.300

### Arms

| arm | task | skill index in the prompt | episodes | mean score |
|---|---|---|---|---|
| `source` | look at pen under the desklamp | (empty) | 8 | 0.713 |
| `check0/control` | look at alarmclock under the desklamp | (empty) | 4 | 0.854 |
| `check0/new0` | look at alarmclock under the desklamp | S1=new0 | 4 | 0.850 |
| `check1/control` | look at box under the desklamp | (empty) | 4 | 0.300 |
| `check1/new0` | look at box under the desklamp | S1=new0 | 4 | 0.679 |

### New skills written on A and their credit

| candidate | name | written by (episode, call) | loaded | sum of usage weights | credit |
|---|---|---|---|---|---|
| new0 | Be careful | [[1, 0], [3, 0], [6, 0]] | 6/8 | 0.00 | +0.000 |

**new0 — Be careful**

```text
When to use: Any household task
1. Read the task carefully.
2. Explore the room.
3. Do not give up.
```

Credit of `new0`, episode by episode (credit = sum of u x (score - baseline) / (sum of u + kappa)):

| check task | episode | loaded | influence | u | score | baseline | u x (score - baseline) |
|---|---|---|---|---|---|---|---|
| B1 | 12 | yes | +0.000 | 0.00 | 0.867 | 0.854 | +0.000 |
| B1 | 13 | yes | +0.000 | 0.00 | 0.800 | 0.854 | -0.000 |
| B1 | 14 | yes | +0.000 | 0.00 | 0.867 | 0.854 | +0.000 |
| B1 | 15 | yes | +0.000 | 0.00 | 0.867 | 0.854 | +0.000 |
| B2 | 20 | yes | +0.000 | 0.00 | 0.683 | 0.300 | +0.000 |
| B2 | 21 | no | - | 0.00 | 0.650 | 0.300 | +0.000 |
| B2 | 22 | no | - | 0.00 | 0.700 | 0.300 | +0.000 |
| B2 | 23 | yes | +0.000 | 0.00 | 0.683 | 0.300 | +0.000 |

### Rewards of the source episodes

Ordinary tokens get the task advantage. The tokens of a write_skill call get the task advantage plus the credit of the skill, normalized over the group (an episode that wrote nothing counts as credit 0).

| episode | score of A | task advantage | wrote | credit | advantage of the write_skill tokens |
|---|---|---|---|---|---|
| 0 | 0.800 | +0.298 | - | - | - |
| 1 | 0.683 | -0.099 | new0@step18 | +0.000 | -0.099 |
| 2 | 0.000 | -2.428 | - | - | - |
| 3 | 0.850 | +0.469 | new0@step8 | +0.000 | +0.469 |
| 4 | 0.850 | +0.469 | - | - | - |
| 5 | 0.817 | +0.355 | - | - | - |
| 6 | 0.850 | +0.469 | new0@step8 | +0.000 | +0.469 |
| 7 | 0.850 | +0.469 | - | - | - |

### Trajectories

**A source episode that wrote a skill** — arm `source`, episode 1: won in 19 steps, reward 0.683, advantage -0.099, advantage on its 51 write_skill tokens -0.099

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

- write_skill: saves what you learned in this task as a new skill for future tasks of the same kind. You may do this at any turn, also right after the task is completed. Reply like this:
Thought: <what is worth saving>
Action: write_skill
Name: <short title>
When to use: <one sentence describing the kind of task>
Procedure:
1. <step>
2. <step>
Write a general procedure: which steps in which order, which command works, how to recover from a typical mistake. It must not depend on this room, so never mention numbered objects such as "drawer 3". Future agents see only the name and the "when to use" line before they decide to load it. Use at most 120 words in the procedure.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a garbagecan 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.

Your task is to: look at pen under the desklamp.

[assistant]
Thought: Let me look somewhere else.
Action: go to garbagecan 1

[user]
You arrive at garbagecan 1. On the garbagecan 1, you see nothing.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 3, a cellphone 2, a laptop 3, a pillow 1, and a teddybear 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a cd 1, a laptop 1, and a pen 1.

[assistant]
Thought: I see pen 1, take it.
Action: take pen 1 from desk 1

[user]
You pick up the pen 1 from the desk 1.

[assistant]
Thought: I have not checked shelf 1 yet.
Action: go to shelf 1

[user]
You arrive at shelf 1. On the shelf 1, you see a alarmclock 1, and a keychain 1.

[assistant]
Thought: Let me look somewhere else.
Action: go to garbagecan 1

[user]
You arrive at garbagecan 1. On the garbagecan 1, you see nothing.

[assistant]
Thought: I have not checked desk 2 yet.
Action: go to desk 2

[user]
You arrive at desk 2. On the desk 2, you see a book 1, a cellphone 1, and a laptop 2.

[assistant]
Thought: Let me look somewhere else.
Action: go to desk 2

[user]
Nothing happens.
Commands available here: examine desk 2, examine pen 1, inventory, look, move pen 1 to desk 2, go to bed 1, go to desk 1, go to garbagecan 1, go to shelf 1, go to shelf 2, go to shelf 3, go to shelf 4, ... (+1 more)

[assistant]
Thought: I have not checked shelf 2 yet.
Action: go to shelf 2

[user]
You arrive at shelf 2. On the shelf 2, you see a creditcard 1, a desklamp 1, and a pencil 1.

[assistant]
Thought: Let me look somewhere else.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a cd 1, and a laptop 1.

[assistant]
Thought: Go back to the lamp.
Action: go to shelf 2

[user]
You arrive at shelf 2. On the shelf 2, you see a creditcard 1, a desklamp 1, and a pencil 1.

[assistant]
Thought: Let me look somewhere else.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a cd 1, and a laptop 1.

[assistant]
Thought: Go back to the lamp.
Action: go to shelf 2

[user]
You arrive at shelf 2. On the shelf 2, you see a creditcard 1, a desklamp 1, and a pencil 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine pen 1, examine shelf 2, inventory, look, move pen 1 to shelf 2, use desklamp 1, go to bed 1, go to desk 1, go to desk 2, go to garbagecan 1, go to shelf 1, go to shelf 3, ... (+2 more)

[assistant]
Thought: Let me look somewhere else.
Action: go to shelf 4

[user]
You arrive at shelf 4. On the shelf 4, you see a pencil 2.

[assistant]
Thought: Go back to the lamp.
Action: go to shelf 2

[user]
You arrive at shelf 2. On the shelf 2, you see a creditcard 1, a desklamp 1, and a pencil 1.

[assistant]
Thought: Try the lamp.
Action: examine pen 1 with desklamp 1

[user]
Nothing happens.
Commands available here: examine pen 1, examine shelf 2, inventory, look, move pen 1 to shelf 2, use desklamp 1, go to bed 1, go to desk 1, go to desk 2, go to garbagecan 1, go to shelf 1, go to shelf 3, ... (+2 more)

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1

[user]
You turn on the desklamp 1.
Task completed. You may save a skill with write_skill, or reply `Action: done`.

[assistant]
Thought: this is worth saving
Action: write_skill
Name: Be careful
When to use: Any household task
Procedure:
1. Read the task carefully.
2. Explore the room.
3. Do not give up.
```

**A check episode with the new skill new0 in its index** — arm `check0/new0`, episode 12: won in 8 steps, reward 0.867, advantage +0.500

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Be careful - Any household task
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: examine the alarmclock with the desklamp.

[assistant]
Thought: maybe this helps.
Action: load_skill S1

[user]
Loaded skill S1:
### Be careful
When to use: Any household task
1. Read the task carefully.
2. Explore the room.
3. Do not give up.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 2, a book 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1, a bowl 1, a creditcard 3, a creditcard 2, a creditcard 1, a keychain 2, a keychain 1, and a laptop 1.

[assistant]
Thought: I see alarmclock 1, take it.
Action: take alarmclock 1 from dresser 1

[user]
You pick up the alarmclock 1 from the dresser 1.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a desklamp 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine alarmclock 1, examine sidetable 1, inventory, look, move alarmclock 1 to sidetable 1, use desklamp 1, go to bed 1, go to drawer 1, go to drawer 10, go to drawer 11, go to drawer 2, go to drawer 3, ... (+10 more)

[assistant]
Thought: Try the lamp.
Action: examine alarmclock 1 with desklamp 1

[user]
Nothing happens.
Commands available here: examine alarmclock 1, examine sidetable 1, inventory, look, move alarmclock 1 to sidetable 1, use desklamp 1, go to bed 1, go to drawer 1, go to drawer 10, go to drawer 11, go to drawer 2, go to drawer 3, ... (+10 more)

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```

**A control episode (old library only)** — arm `check0/control`, episode 8: won in 7 steps, reward 0.883, advantage +0.789

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: examine the alarmclock with the desklamp.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 2, a book 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a alarmclock 1, a bowl 1, a creditcard 3, a creditcard 2, a creditcard 1, a keychain 2, a keychain 1, and a laptop 1.

[assistant]
Thought: I see alarmclock 1, take it.
Action: take alarmclock 1 from dresser 1

[user]
You pick up the alarmclock 1 from the dresser 1.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a desklamp 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine alarmclock 1, examine sidetable 1, inventory, look, move alarmclock 1 to sidetable 1, use desklamp 1, go to bed 1, go to drawer 1, go to drawer 10, go to drawer 11, go to drawer 2, go to drawer 3, ... (+10 more)

[assistant]
Thought: Try the lamp.
Action: examine alarmclock 1 with desklamp 1

[user]
Nothing happens.
Commands available here: examine alarmclock 1, examine sidetable 1, inventory, look, move alarmclock 1 to sidetable 1, use desklamp 1, go to bed 1, go to drawer 1, go to drawer 10, go to drawer 11, go to drawer 2, go to drawer 3, ... (+10 more)

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```


---

## Unit 4 — a merge of similar library skills is drafted and judged

Similar skills of one task type are merged into one draft. The draft replaces them only if its credit is at least the credit of the skills it replaces, measured against the index without any of them.

Step 2, task type `look`, library version 1 (uid `b1df6b4c-9e2b-4f46-a048-237744d325fd`).

- A (source task): `look at bowl under the desklamp`
- B1 (check task): `look at pen under the desklamp`, baseline score 0.879
- B2 (check task): `look at mug under the desklamp`, baseline score 0.900

### Arms

| arm | task | skill index in the prompt | episodes | mean score |
|---|---|---|---|---|
| `source` | look at bowl under the desklamp | S1=sk0001, S2=sk0002, S3=sk0000 | 8 | 0.875 |
| `check0/control` | look at pen under the desklamp | S1=sk0001, S2=sk0002, S3=sk0000 | 4 | 0.879 |
| `check0/new0` | look at pen under the desklamp | S1=sk0001, S2=sk0002, S3=sk0000, S4=new0 | 4 | 0.883 |
| `check0/new1` | look at pen under the desklamp | S1=new1, S2=sk0001, S3=sk0002, S4=sk0000 | 4 | 0.787 |
| `check0/merge/without` | look at pen under the desklamp | S1=sk0000 | 4 | 0.854 |
| `check0/merge/members` | look at pen under the desklamp | S1=sk0001, S2=sk0000, S3=sk0002 | 4 | 0.900 |
| `check0/merge/merged` | look at pen under the desklamp | S1=sk0000, S2=merged | 4 | 0.892 |
| `check1/control` | look at mug under the desklamp | S1=sk0001, S2=sk0002, S3=sk0000 | 4 | 0.900 |
| `check1/new0` | look at mug under the desklamp | S1=sk0001, S2=sk0002, S3=sk0000, S4=new0 | 4 | 0.908 |
| `check1/new1` | look at mug under the desklamp | S1=sk0001, S2=new1, S3=sk0002, S4=sk0000 | 4 | 0.912 |
| `check1/merge/without` | look at mug under the desklamp | S1=sk0000 | 4 | 0.883 |
| `check1/merge/members` | look at mug under the desklamp | S1=sk0000, S2=sk0001, S3=sk0002 | 4 | 0.892 |
| `check1/merge/merged` | look at mug under the desklamp | S1=merged, S2=sk0000 | 4 | 0.633 |

### New skills written on A and their credit

| candidate | name | written by (episode, call) | loaded | sum of usage weights | credit |
|---|---|---|---|---|---|
| new0 | Examine under a lamp | [[0, 0]] | 1/8 | 1.00 | -0.008 |
| new1 | Examine under a lamp | [[1, 0], [5, 0]] | 2/8 | 2.00 | -0.054 |

**new1 — Examine under a lamp**

```text
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
```

Credit of `new1`, episode by episode (credit = sum of u x (score - baseline) / (sum of u + kappa)):

| check task | episode | loaded | influence | u | score | baseline | u x (score - baseline) |
|---|---|---|---|---|---|---|---|
| B1 | 16 | no | - | 0.00 | 0.883 | 0.879 | +0.000 |
| B1 | 17 | yes | +2.000 | 1.00 | 0.700 | 0.879 | -0.179 |
| B1 | 18 | no | - | 0.00 | 0.800 | 0.879 | -0.000 |
| B1 | 19 | no | - | 0.00 | 0.767 | 0.879 | -0.000 |
| B2 | 40 | no | - | 0.00 | 0.917 | 0.900 | +0.000 |
| B2 | 41 | no | - | 0.00 | 0.917 | 0.900 | +0.000 |
| B2 | 42 | yes | +0.500 | 1.00 | 0.917 | 0.900 | +0.017 |
| B2 | 43 | no | - | 0.00 | 0.900 | 0.900 | +0.000 |

### Rewards of the source episodes

Ordinary tokens get the task advantage. The tokens of a write_skill call get the task advantage plus the credit of the skill, normalized over the group (an episode that wrote nothing counts as credit 0).

| episode | score of A | task advantage | wrote | credit | advantage of the write_skill tokens |
|---|---|---|---|---|---|
| 0 | 0.883 | +0.418 | new0@step6 | -0.008 | +0.672 |
| 1 | 0.867 | -0.418 | new1@step7 | -0.054 | -2.027 |
| 2 | 0.833 | -2.092 | - | - | - |
| 3 | 0.883 | +0.418 | - | - | - |
| 4 | 0.867 | -0.418 | - | - | - |
| 5 | 0.883 | +0.418 | new1@step6 | -0.054 | -1.191 |
| 6 | 0.883 | +0.418 | - | - | - |
| 7 | 0.900 | +1.255 | - | - | - |

**Merge of ['sk0001', 'sk0002']**: credit of the merged skill +0.030, credit of the original skills +0.036.

### Trajectories

**A source episode that wrote a skill** — arm `source`, episode 0: won in 7 steps, reward 0.883, advantage +0.418, advantage on its 106 write_skill tokens +0.672

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
[S2] Examine under a lamp - The task is to look at an object under a lamp
[S3] Find, take, place - The task is to put an object in or on a receptacle
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

- write_skill: saves what you learned in this task as a new skill for future tasks of the same kind. You may do this at any turn, also right after the task is completed. Reply like this:
Thought: <what is worth saving>
Action: write_skill
Name: <short title>
When to use: <one sentence describing the kind of task>
Procedure:
1. <step>
2. <step>
Write a general procedure: which steps in
... [395 more characters]

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, and a shelf 1.

Your task is to: look at bowl under the desklamp.

[assistant]
Thought: this skill is about my task.
Action: load_skill S2

[user]
Loaded skill S2:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
4. If the game rejects a command, read the commands it lists as available and use one of those.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a laptop 2, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a book 1, a bowl 1, a cd 1, a creditcard 1, and a mug 1.

[assistant]
Thought: I see bowl 1, take it.
Action: take bowl 1 from desk 1

[user]
You pick up the bowl 1 from the desk 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a desklamp 1.

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1

[user]
You turn on the desklamp 1.
Task completed. You may save a skill with write_skill, or reply `Action: done`.

[assistant]
Thought: this is worth saving
Action: write_skill
Name: Examine under a lamp
When to use: The task is to look at an object under a lamp
Procedure:
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
4. If the game rejects a command, read the commands it lists as available and use one of those.
```

**A check episode with the new skill new1 in its index** — arm `check0/new1`, episode 17: won in 18 steps, reward 0.700, advantage -1.148

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
[S2] Examine under a lamp - The task is to look at an object under a lamp
[S3] Examine under a lamp - The task is to look at an object under a lamp
[S4] Find, take, place - The task is to put an object in or on a receptacle
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 3, a armchair 2, a armchair 1, a bed 1, a cabinet 2, a cabinet 1, a desk 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: examine the pen with the desklamp.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 3, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 1, a creditcard 2, a houseplant 1, and a mug 1.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a cellphone 2, a creditcard 1, a desklamp 1, a pen 2, a pen 1, and a remotecontrol 1.

[assistant]
Thought: Let me look somewhere else.
Action: go to cabinet 1

[user]
You arrive at cabinet 1. The cabinet 1 is closed.

[assistant]
Thought: Let me look somewhere else.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 3, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked armchair 1 yet.
Action: go to armchair 1

[user]
You arrive at armchair 1. On the armchair 1, you see nothing.

[assistant]
Thought: I have not checked garbagecan 1 yet.
Action: go to garbagecan 1

[user]
You arrive at garbagecan 1. On the garbagecan 1, you see a pencil 1.

[assistant]
Thought: I have not checked armchair 2 yet.
Action: go to armchair 2

[user]
You arrive at armchair 2. On the armchair 2, you see a book 2.

[assistant]
Thought: I have not checked armchair 3 yet.
Action: go to armchair 3

[user]
You arrive at armchair 3. On the armchair 3, you see a keychain 2.

[assistant]
Thought: I have not checked laundryhamper 1 yet.
Action: go to laundryhamper 1

[user]
You arrive at laundryhamper 1. On the laundryhamper 1, you see nothing.

[assistant]
Thought: Let me look somewhere else.
Action: go to armchair 1

[user]
You arrive at armchair 1. On the armchair 1, you see nothing.

[assistant]
Thought: I have not checked safe 1 yet.
Action: go to safe 1

[user]
You arrive at safe 1. On the safe 1, you see a cellphone 1.

[assistant]
Thought: I have not checked drawer 1 yet.
Action: go to drawer 1

[user]
You arrive at drawer 1. The drawer 1 is closed.

[assistant]
Thought: Let me look somewhere else.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a cellphone 2, a creditcard 1, a desklamp 1, a pen 2, a pen 1, and a remotecontrol 1.

[assistant]
Thought: I see pen 2, take it.
Action: take pen 2 from sidetable 1

[user]
You pick up the pen 2 from the sidetable 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine pen 2, examine sidetable 1, inventory, look, move pen 2 to sidetable 1, use desklamp 1, go to armchair 1, go to armchair 2, go to armchair 3, go to bed 1, go to cabinet 1, go to cabinet 2, ... (+8 more)

[assistant]
Thought: I am stuck, a skill may help.
Action: load_skill S1

[user]
Loaded skill S1:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.

Continue with the task.

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```

**A check episode that did not use new1 (usage weight 0)** — arm `check0/new1`, episode 16: won in 7 steps, reward 0.883, advantage +1.257

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
[S2] Examine under a lamp - The task is to look at an object under a lamp
[S3] Examine under a lamp - The task is to look at an object under a lamp
[S4] Find, take, place - The task is to put an object in or on a receptacle
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 3, a armchair 2, a armchair 1, a bed 1, a cabinet 2, a cabinet 1, a desk 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: examine the pen with the desklamp.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 3, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 1, a creditcard 2, a houseplant 1, and a mug 1.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a cellphone 2, a creditcard 1, a desklamp 1, a pen 2, a pen 1, and a remotecontrol 1.

[assistant]
Thought: I see pen 2, take it.
Action: take pen 2 from sidetable 1

[user]
You pick up the pen 2 from the sidetable 1.

[assistant]
Thought: Try the lamp.
Action: turn on desklamp 1

[user]
Nothing happens.
Commands available here: examine pen 2, examine sidetable 1, inventory, look, move pen 2 to sidetable 1, use desklamp 1, go to armchair 1, go to armchair 2, go to armchair 3, go to bed 1, go to cabinet 1, go to cabinet 2, ... (+8 more)

[assistant]
Thought: I am stuck, a skill may help.
Action: load_skill S2

[user]
Loaded skill S2:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.

Continue with the task.

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```

**A control episode (old library only)** — arm `check0/control`, episode 8: won in 6 steps, reward 0.900, advantage +0.833

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
[S2] Examine under a lamp - The task is to look at an object under a lamp
[S3] Find, take, place - The task is to put an object in or on a receptacle
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a armchair 3, a armchair 2, a armchair 1, a bed 1, a cabinet 2, a cabinet 1, a desk 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, and a sidetable 1.

Your task is to: examine the pen with the desklamp.

[assistant]
Thought: this skill is about my task.
Action: load_skill S1

[user]
Loaded skill S1:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a cellphone 3, a laptop 1, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a alarmclock 1, a creditcard 2, a houseplant 1, and a mug 1.

[assistant]
Thought: I have not checked sidetable 1 yet.
Action: go to sidetable 1

[user]
You arrive at sidetable 1. On the sidetable 1, you see a cellphone 2, a creditcard 1, a desklamp 1, a pen 2, a pen 1, and a remotecontrol 1.

[assistant]
Thought: I see pen 2, take it.
Action: take pen 2 from sidetable 1

[user]
You pick up the pen 2 from the sidetable 1.

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```


---

## Unit 5 — validation with the frozen library

Validation only reads the library: skills are retrieved and may be loaded, nothing is written or staged.

Step 8, task type `look`, library version 8 (uid `35bd844d-6b2d-4e20-a71c-53038e51b207`).

- A (source task): `look at keychain under the desklamp`

### Arms

| arm | task | skill index in the prompt | episodes | mean score |
|---|---|---|---|---|
| `eval` | look at keychain under the desklamp | S1=sk0001, S2=sk0000, S3=sk0002, S4=sk0003 | 1 | 0.900 |

### Trajectories

**A validation episode** — arm `eval`, episode 0: won in 6 steps, reward 0.900, advantage -

```text
[system]
You are an agent acting in a text-based household environment. Each turn you get an observation and reply with one command.

Commands you can use (replace the <...> parts with names that appeared in an observation, e.g. "mug 1"):
- go to <receptacle>
- open <receptacle> | close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> | cool <object> with <receptacle> | clean <object> with <receptacle>
- use <object>
- examine <object> | look | inventory
You only see what is at your current location, and you must be at a receptacle to interact with it.

## Skills
A skill is a short reusable procedure that an agent wrote down during an earlier task. Skill commands take one step, like any other command.

Skills you can load (only the name and the "when to use" are listed, the procedure appears after loading):
[S1] Examine under a lamp - The task is to look at an object under a lamp
[S2] Find, take, place - The task is to put an object in or on a receptacle
[S3] Examine under a lamp - The task is to look at an object under a lamp
[S4] Heat with the microwave, then place - The task is to heat an object and put it somewhere
- load_skill S<number>: shows the procedure of that skill. Load a skill when its "when to use" matches your task.

Reply in exactly this format:
Thought: <one short sentence about what to do next>
Action: <one command>

[user]
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.

Your task is to: examine the keychain with the desklamp.

[assistant]
Thought: this skill is about my task.
Action: load_skill S3

[user]
Loaded skill S3:
### Examine under a lamp
When to use: The task is to look at an object under a lamp
1. Search receptacles until you see the object and take it.
2. Keep holding it and find a desklamp or floorlamp.
3. Turn the lamp on with: use <lamp>.
4. If the game rejects a command, read the commands it lists as available and use one of those.

Continue with the task.

[assistant]
Thought: I have not checked bed 1 yet.
Action: go to bed 1

[user]
You arrive at bed 1. On the bed 1, you see a book 2, a cellphone 3, a cellphone 2, a pillow 2, and a pillow 1.

[assistant]
Thought: I have not checked desk 1 yet.
Action: go to desk 1

[user]
You arrive at desk 1. On the desk 1, you see a book 1, and a laptop 1.

[assistant]
Thought: I have not checked dresser 1 yet.
Action: go to dresser 1

[user]
You arrive at dresser 1. On the dresser 1, you see a cd 3, a cellphone 1, a desklamp 1, a keychain 3, a pencil 3, and a pencil 2.

[assistant]
Thought: I see keychain 3, take it.
Action: take keychain 3 from dresser 1

[user]
You pick up the keychain 3 from the dresser 1.

[assistant]
Thought: Turn the lamp on while holding the object.
Action: use desklamp 1
```

