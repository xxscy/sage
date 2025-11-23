# Human Interaction Tool 调用分析报告

## 统计摘要

### 总体统计
- **总案例数**: 30
- **正确预测**: 4 (13.3%)
- **错误预测**: 26 (86.7%)

### 参数类别分布
- **用户偏好**: 29 次
- **模糊术语**: 25 次
- **缺失信息**: 19 次
- **代词歧义**: 16 次
- **设备歧义**: 8 次

### 设备类型分布
- **TV**: 15 次
- **light**: 8 次
- **dishwasher**: 1 次
- **fridge**: 1 次
- **living room**: 1 次
- **dining room**: 1 次

### 动作类型分布
- **put**: 5 次
- **turn on**: 4 次
- **set**: 3 次
- **change**: 3 次
- **turn off**: 3 次
- **play**: 3 次
- **adjust**: 1 次
- **make**: 1 次
- **dim**: 1 次


## 详细表格

| 序号 | 测试名称 | 用户命令 | 用户名 | 测试类型 | 设备类型 | 动作类型 | 参数类别 | 缺失信息 | 模糊术语 | 调用原因 | 是否正确 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | getting_call_tv_too_loud | Amal : I am getting a call, adjust the volume of t... | amal | intent_resolution, device_resolution | TV | adjust | 缺失信息, 模糊术语, 用户偏好, 代词歧义 | the desired volume level; the desired volume level... | pronouns or unclear references | The command lacks specific information about the desired volume level, which can... | 否 |
| 2 | dishes_dirty_set_appropriate_mode | Amal : Dishes are too greasy, set an appropriate m... | amal | intent_resolution | dishwasher | set | 缺失信息, 用户偏好 | what "appropriate mode" means for washing greasy d... | - | The command lacks specific information about what "appropriate mode" means for w... | 否 |
| 3 | put_something_informative | Amal : Put something informative on the tv by the ... | amal | intent_resolution, personalization, device_resolution | TV | put | 缺失信息, 模糊术语, 用户偏好 | what type of content is desired, which introduces ... | the type of informative content to display on the ... | The command contains ambiguous wording regarding the type of informative content... | 否 |
| 4 | change_light_colors_conditioned_on_favourite_team | Abhisek : Change the lights of the house to repres... | abhisek | intent_resolution, personalization, device_resolution, command_chaining | TV | change | 缺失信息, 用户偏好, 代词歧义 | the retrieved data; what colors or settings should... | - | The command lacks critical information about the specific colors or settings to ... | 否 |
| 5 | set_bedroom_light_for_sleeping | Amal : I am going to sleep. Change the bedroom lig... | amal | intent_resolution, personalization, device_resolution | light | change | 缺失信息, 用户偏好 | how to change the bedroom light, and there is no a... | - | The command lacks specific details on how to change the bedroom light, and there... | 否 |
| 6 | turn_off_tvs_turn_on_fireplace_light | Dmitriy : turn off all the TVs and switch on the f... | dmitriy | device_resolution, command_chaining | TV | turn off | 模糊术语, 用户偏好, 代词歧义 | - | references regarding the fireplace light, which la... | The command contains ambiguous references regarding the fireplace light, which l... | 否 |
| 7 | control_fridge_temp | Abhisek : Change the fridge internal temperature t... | abhisek | simple | fridge | change | 缺失信息, 模糊术语, 用户偏好, 代词歧义 | the available data, clarification is needed | pronouns or unclear intent | The command lacks critical user preference information regarding the fridge temp... | 否 |
| 8 | frige_door_light | abhisek : turn on the light in the dining room whe... | abhisek | persistence | light | turn on | 模糊术语, 用户偏好 | - | references regarding which light to turn on, and c... | The command contains ambiguous references regarding which light to turn on, and ... | 否 |
| 9 | turn_on_all_lights | Abhisek : Turn on all the lights. | abhisek | simple | light | turn on | 模糊术语, 用户偏好 | - | references regarding which lights to turn on, and ... | The command contains ambiguous references regarding which lights to turn on, and... | 否 |
| 10 | turn_on_bedroom_light_dishwasherstate | abhisek : turn on light by the nightstand when the... | abhisek | persistence, device_resolution | light | turn on | 模糊术语, 用户偏好, 设备歧义 | - | references regarding which light to turn on, and c... | The command contains ambiguous references regarding which light to turn on, and ... | 否 |
| 11 | increase_volume_with_dishwasher_on | abhisek : increase the volume of the TV by the cre... | abhisek | persistence, device_resolution | TV | - | 缺失信息, 模糊术语, 用户偏好, 设备歧义, 代词歧义 | the desired volume level to increase to | pronouns or unclear references, but it lacks infor... | The command lacks critical user preferences regarding the volume level for the T... | 否 |
| 12 | put_the_game_on_amal | amal : put the game on the tv by the credenza | amal | intent_resolution, personalization, device_resolution | TV | put | 缺失信息, 模糊术语, 用户偏好 | which game the user wants to watch, and there are ... | references regarding which game to display, requir... | The command contains ambiguous references regarding which game to display, requi... | 否 |
| 13 | set_bedroom_light_fav_color | Amal : Set bedroom light to my favourite color | amal | intent_resolution, human_interaction, device_resolution | light | set | 缺失信息, 模糊术语, 用户偏好, 代词歧义 | Amal's favorite color in the provided preferences ... | my favourite color,; references to "my favourite c... | The command contains ambiguous references to "my favourite color," which is not ... | 是 |
| 14 | put_the_game_on_abhisek | abhisek : put the game on the tv by the credenza | abhisek | intent_resolution, personalization, device_resolution | TV | put | 缺失信息, 模糊术语, 用户偏好 | the available data; which game to put on | references to "the game" without specifying which ... | The command contains ambiguous references to "the game" without specifying which... | 否 |
| 15 | put_the_game_on_dmitriy | dmitriy : put the game on the tv by the credenza | dmitriy | intent_resolution, human_interaction, personalization, device_resolution | TV | put | 缺失信息, 模糊术语, 用户偏好, 代词歧义 | which game is being referred to | the game,; the game | The command contains an ambiguous reference to "the game," which requires clarif... | 是 |
| 16 | long_day_unwind | dmitriy : its been a long, tiring day. Can you pla... | dmitriy | intent_resolution, personalization, device_resolution | TV | play | 缺失信息, 模糊术语, 用户偏好, 设备歧义, 代词歧义 | a particular show or movie; what content is availa... | content references that need clarification regardi... | The command contains ambiguous content references that need clarification regard... | 否 |
| 17 | redonkulous_living_room | Dmitriy : Make the living room look redonkulous! | dmitriy | intent_resolution, human_interaction, device_resolution | living room | make | 用户偏好, 代词歧义 | - | - | The command contains non-standard expressions and lacks critical user preference... | 是 |
| 18 | switch_off_everything | Abhisek : Heading off to work. Turn off all the no... | abhisek | intent_resolution, device_resolution, command_chaining | - | turn off | 缺失信息, 模糊术语, 用户偏好, 设备歧义 | what devices Abhisek considers non-essential | non essential devices; references to "non essentia... | The command contains ambiguous references to "non essential devices" which need ... | 否 |
| 19 | turn_it_off | Abhisek : Turn it off. | abhisek | intent_resolution, human_interaction, device_resolution | - | - | 模糊术语, 用户偏好, 设备歧义, 代词歧义 | - | pronouns and needs clarification on which device t... | The command contains ambiguous pronouns and needs clarification on which device ... | 是 |
| 20 | room_too_bright | Amal : It is too bright in the dining room. | amal | intent_resolution, device_resolution | dining room | - | 缺失信息, 模糊术语, 用户偏好, 代词歧义 | the available data, clarification is needed | pronouns or unclear intent; it clearly states the ... | The command lacks critical user preferences for lighting in the dining room, whi... | 否 |
| 21 | dim_fireplace_lamp | Abhisek : dim the lights by the fire place to a th... | abhisek | intent_resolution, device_resolution | light | dim | 缺失信息, 模糊术语, 用户偏好 | which lights are to be dimmed, as there may be mul... | references regarding which lights to dim, requirin... | The command contains ambiguous references regarding which lights to dim, requiri... | 否 |
| 22 | tv_off_lights_on_persist | Amal: when the TV  by the credenza turns off turn ... | amal | persistence, device_resolution | TV | turn on | 缺失信息, 模糊术语, 用户偏好 | which light to turn on, as there may be multiple l... | references regarding which light to turn on, requi... | The command contains ambiguous references regarding which light to turn on, requ... | 否 |
| 23 | lower_tv_volume | Amal : Lower the volume of the TV by the light | amal | device_resolution | TV | - | 模糊术语, 用户偏好 | - | phrasing that requires clarification regarding the... | The command contains ambiguous phrasing that requires clarification regarding th... | 否 |
| 24 | switch_to_other_tv | Amal: move this channel to the other TV and turn t... | amal | intent_resolution, device_resolution, command_chaining | TV | - | 模糊术语, 用户偏好, 设备歧义, 代词歧义 | - | references with "this channel" and "this one," as ... | The command contains ambiguous references that need clarification regarding whic... | 否 |
| 25 | put_the_game_on_dim_the_lights | Amal: put the game on the tv by the credenza and d... | amal | intent_resolution, personalization, device_resolution, command_chaining | TV | put | 缺失信息, 模糊术语, 用户偏好 | which lights to dim | references to the TV and lights that require clari... | The command contains ambiguous references to the TV and lights that require clar... | 否 |
| 26 | memory_weather_test | Abhisek : I am going to visit my mom. Should I bri... | abhisek | personalization | - | - | 模糊术语, 用户偏好, 代词歧义 | - | pronouns or unclear references; however, it lacks ... | The command lacks critical user preferences regarding weather, which cannot be r... | 否 |
| 27 | play_something_for_kids | Abhisek : play something for the kids on the TV by... | abhisek | intent_resolution, device_resolution | TV | play | 缺失信息, 模糊术语, 用户偏好, 设备歧义 | what "something" refers to, particularly regarding... | references regarding the type of content suitable ... | The command contains ambiguous references regarding the type of content suitable... | 否 |
| 28 | put_on_something_funny | Dmitriy : play something funny on the TV by the pl... | dmitriy | intent_resolution, device_resolution | TV | play | 模糊术语, 用户偏好, 代词歧义 | - | references regarding what "something funny" refers... | The command contains ambiguous references regarding what "something funny" refer... | 否 |
| 29 | setup_lights_for_dinner | Dmitriy : set up lights for dinner | dmitriy | intent_resolution, device_resolution | light | set | 缺失信息, 用户偏好, 代词歧义 | what kind of lighting is preferred for dinner, lea... | - | The command lacks specific lighting preferences for dinner, which cannot be reli... | 否 |
| 30 | turn_off_light_dim | Dmitriy : turn off all the lights that are dim | dmitriy | device_resolution, command_chaining | light | turn off | 模糊术语, 设备歧义, 代词歧义 | - | references to "all the lights that are dim," and s... | The command contains ambiguous references to "all the lights that are dim," and ... | 否 |

## LLM调用human_interaction_tool的主要原因


根据分析，LLM提出需要调用human_interaction_tool的主要原因包括：

### 1. **信息缺失 (Missing Information)**
- **描述**: 命令中缺少执行操作所需的关键信息
- **示例**: 
  - "adjust the volume" - 缺少具体音量值
  - "set an appropriate mode" - 缺少模式定义
  - "my favourite color" - 缺少颜色偏好信息

### 2. **模糊术语 (Ambiguous Terms)**
- **描述**: 命令中包含不明确或主观的术语
- **示例**:
  - "the game" - 未指定具体游戏
  - "redonkulous" - 非标准主观表达
  - "non essential devices" - 未定义哪些设备

### 3. **代词歧义 (Pronoun Ambiguity)**
- **描述**: 使用代词但上下文不明确
- **示例**:
  - "Turn it off" - "it"指代不明
  - "this channel" - "this"指代不明

### 4. **用户偏好缺失 (Missing User Preferences)**
- **描述**: 需要用户偏好但系统中不存在
- **示例**:
  - "my favourite color" - 偏好未存储
  - "cozy setting" - 偏好定义不明确

### 5. **设备歧义 (Device Ambiguity)**
- **描述**: 设备识别不明确
- **示例**:
  - "the TV" - 多个TV时指代不明
  - "the light" - 多个灯时指代不明

### 6. **非标准表达 (Non-standard Expressions)**
- **描述**: 使用非标准或创造性词汇
- **示例**:
  - "redonkulous" - 非标准词汇
  - "something light and entertaining" - 主观描述

### 调用决策流程

LLM通过以下步骤决定是否调用human_interaction_tool：

1. **关键词分析**: 识别命令中的动作和目标
2. **歧义检查**: 检测模糊术语、代词、非标准表达
3. **信息可用性检查**: 验证所需信息是否可从用户偏好、历史记录或设备状态中获取
4. **决策**: 如果关键信息缺失且无法推断，则调用human_interaction_tool

### 最佳实践建议

1. **增强上下文理解**: 利用对话历史解决代词歧义
2. **完善用户偏好库**: 存储更多用户偏好以减少询问
3. **改进设备识别**: 使用更智能的设备匹配算法
4. **处理非标准表达**: 建立同义词和上下文映射
5. **智能推断**: 在安全范围内进行合理推断，减少不必要的交互
