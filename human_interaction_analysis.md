# Human Interaction Tool 调用分析

## 总览

- 总案例数: 30
- 正确预测: 4
- 错误预测: 26

## 详细表格

| 序号 | 测试名称 | 用户命令 | 用户名 | 测试类型 | 设备类型 | 动作类型 | 缺失信息 | 模糊术语 | 调用原因 | 是否正确 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | getting_call_tv_too_loud | Amal : I am getting a call, adjust the volume of t... | amal | intent_resolution, device_resolution | TV | adjust | the desired volume level; the desired vo... | adjust; volume of the TV.; volume | The command lacks specific information about the desired vol... | ✗ |
| 2 | dishes_dirty_set_appropriate_mode | Amal : Dishes are too greasy, set an appropriate m... | amal | intent_resolution | dishwasher | set | what "appropriate mode" means for washin... | set; appropriate mode,; dishwasher. | The command lacks specific information about what "appropria... | ✗ |
| 3 | put_something_informative | Amal : Put something informative on the tv by the ... | amal | intent_resolution, personalization, device_resolution | TV |  |  | put,; something informative,; tv by the ... | The command contains ambiguous wording regarding the type of... | ✗ |
| 4 | change_light_colors_conditioned_on_favourite_team | Abhisek : Change the lights of the house to repres... | abhisek | intent_resolution, personalization, device_resolution, command_chaining | TV | change | the retrieved data |  | The command lacks critical information about the specific co... | ✗ |
| 5 | set_bedroom_light_for_sleeping | Amal : I am going to sleep. Change the bedroom lig... | amal | intent_resolution, personalization, device_resolution | light | change |  | change; bedroom light,; change the bedro... | The command lacks specific details on how to change the bedr... | ✗ |
| 6 | turn_off_tvs_turn_on_fireplace_light | Dmitriy : turn off all the TVs and switch on the f... | dmitriy | device_resolution, command_chaining | TV | turn off |  | turn off all the TVs; switch on the fire... | The command contains ambiguous references regarding the fire... | ✗ |
| 7 | control_fridge_temp | Abhisek : Change the fridge internal temperature t... | abhisek | simple | fridge | change | the available data | Change,; fridge internal temperature,; 5... | The command lacks critical user preference information regar... | ✗ |
| 8 | frige_door_light | abhisek : turn on the light in the dining room whe... | abhisek | persistence | fridge | turn on |  | turn on; open,; light | The command contains ambiguous references regarding which li... | ✗ |
| 9 | turn_on_all_lights | Abhisek : Turn on all the lights. | abhisek | simple | light | turn on |  | Turn on all the lights; turn on; all the... | The command contains ambiguous references regarding which li... | ✗ |
| 10 | turn_on_bedroom_light_dishwasherstate | abhisek : turn on light by the nightstand when the... | abhisek | persistence, device_resolution | dishwasher | turn on |  | turn on; light, | The command contains ambiguous references regarding which li... | ✗ |
| 11 | increase_volume_with_dishwasher_on | abhisek : increase the volume of the TV by the cre... | abhisek | persistence, device_resolution | TV |  | the desired volume level to increase to | volume; volume | The command lacks critical user preferences regarding the vo... | ✗ |
| 12 | put_the_game_on_amal | amal : put the game on the tv by the credenza | amal | intent_resolution, personalization, device_resolution | TV |  |  | put; the game on the tv by the credenza. | The command contains ambiguous references regarding which ga... | ✗ |
| 13 | set_bedroom_light_fav_color | Amal : Set bedroom light to my favourite color | amal | intent_resolution, human_interaction, device_resolution | light | set | Amal's favorite color in the provided pr... | bedroom light; my favourite color,; my f... | The command contains ambiguous references to "my favourite c... | ✓ |
| 14 | put_the_game_on_abhisek | abhisek : put the game on the tv by the credenza | abhisek | intent_resolution, personalization, device_resolution | TV |  | the available data | put; the game; the TV by the credenza. | The command contains ambiguous references to "the game" with... | ✗ |
| 15 | put_the_game_on_dmitriy | dmitriy : put the game on the tv by the credenza | dmitriy | intent_resolution, human_interaction, personalization, device_resolution | TV |  |  | put; the game; the tv by the credenza. | The command contains an ambiguous reference to "the game," w... | ✓ |
| 16 | long_day_unwind | dmitriy : its been a long, tiring day. Can you pla... | dmitriy | intent_resolution, personalization, device_resolution | TV | play |  | play,; light; entertaining, | The command contains ambiguous content references that need ... | ✗ |
| 17 | redonkulous_living_room | Dmitriy : Make the living room look redonkulous! | dmitriy | intent_resolution, human_interaction, device_resolution |  |  |  | make,; living room,; redonkulous, | The command contains non-standard expressions and lacks crit... | ✓ |
| 18 | switch_off_everything | Abhisek : Heading off to work. Turn off all the no... | abhisek | intent_resolution, device_resolution, command_chaining |  | turn off |  | Turn off; all the non essential devices.... | The command contains ambiguous references to "non essential ... | ✗ |
| 19 | turn_it_off | Abhisek : Turn it off. | abhisek | intent_resolution, human_interaction, device_resolution |  |  |  | Turn it off; turn; it, | The command contains ambiguous pronouns and needs clarificat... | ✓ |
| 20 | room_too_bright | Amal : It is too bright in the dining room. | amal | intent_resolution, device_resolution |  |  | the available data | too bright; dining room, | The command lacks critical user preferences for lighting in ... | ✗ |
| 21 | dim_fireplace_lamp | Abhisek : dim the lights by the fire place to a th... | abhisek | intent_resolution, device_resolution | light | dim |  | dim,; lights,; by the fire place, | The command contains ambiguous references regarding which li... | ✗ |
| 22 | tv_off_lights_on_persist | Amal: when the TV  by the credenza turns off turn ... | amal | persistence, device_resolution | TV | turn on | which light to turn on | by the credenza,; by the bed. | The command contains ambiguous references regarding which li... | ✗ |
| 23 | lower_tv_volume | Amal : Lower the volume of the TV by the light | amal | device_resolution | TV |  |  | by the light; by the light | The command contains ambiguous phrasing that requires clarif... | ✗ |
| 24 | switch_to_other_tv | Amal: move this channel to the other TV and turn t... | amal | intent_resolution, device_resolution, command_chaining | TV |  |  | move; turn off,; this channel | The command contains ambiguous references that need clarific... | ✗ |
| 25 | put_the_game_on_dim_the_lights | Amal: put the game on the tv by the credenza and d... | amal | intent_resolution, personalization, device_resolution, command_chaining | TV | dim | which lights to dim | put the game on the TV; dim the lights b... | The command contains ambiguous references to the TV and ligh... | ✗ |
| 26 | memory_weather_test | Abhisek : I am going to visit my mom. Should I bri... | abhisek | personalization |  |  |  |  | The command lacks critical user preferences regarding weathe... | ✗ |
| 27 | play_something_for_kids | Abhisek : play something for the kids on the TV by... | abhisek | intent_resolution, device_resolution | TV | play |  | play,; something for the kids,; TV | The command contains ambiguous references regarding the type... | ✗ |
| 28 | put_on_something_funny | Dmitriy : play something funny on the TV by the pl... | dmitriy | intent_resolution, device_resolution | TV | play |  | play,; funny,; TV | The command contains ambiguous references regarding what "so... | ✗ |
| 29 | setup_lights_for_dinner | Dmitriy : set up lights for dinner | dmitriy | intent_resolution, device_resolution | light | set |  | color | The command lacks specific lighting preferences for dinner, ... | ✗ |
| 30 | turn_off_light_dim | Dmitriy : turn off all the lights that are dim | dmitriy | device_resolution, command_chaining | light | turn off |  | turn off; all the lights that are dim.; ... | The command contains ambiguous references to "all the lights... | ✗ |

## 调用原因分类

### 原因统计

- **The command contains ambiguous references regardin**: 9 次
- **The command lacks critical user preferences regard**: 2 次
- **The command lacks specific information about the d**: 1 次
- **The command lacks specific information about what **: 1 次
- **The command contains ambiguous wording regarding t**: 1 次
- **The command lacks critical information about the s**: 1 次
- **The command lacks specific details on how to chang**: 1 次
- **The command lacks critical user preference informa**: 1 次
- **The command contains ambiguous references to "my f**: 1 次
- **The command contains ambiguous references to "the **: 1 次
- **The command contains an ambiguous reference to "th**: 1 次
- **The command contains ambiguous content references **: 1 次
- **The command contains non-standard expressions and **: 1 次
- **The command contains ambiguous references to "non **: 1 次
- **The command contains ambiguous pronouns and needs **: 1 次
- **The command lacks critical user preferences for li**: 1 次
- **The command contains ambiguous phrasing that requi**: 1 次
- **The command contains ambiguous references that nee**: 1 次
- **The command contains ambiguous references to the T**: 1 次
- **The command lacks specific lighting preferences fo**: 1 次
- **The command contains ambiguous references to "all **: 1 次

### 设备类型分布

- **TV**: 15 次
- **light**: 6 次
- **未指定**: 5 次
- **dishwasher**: 2 次
- **fridge**: 2 次

### 动作类型分布

- **未指定**: 11 次
- **turn on**: 4 次
- **set**: 3 次
- **change**: 3 次
- **turn off**: 3 次
- **play**: 3 次
- **dim**: 2 次
- **adjust**: 1 次
