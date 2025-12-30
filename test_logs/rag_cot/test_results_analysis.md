# 表格1：总体统计和类型正确率

| 测试运行 | 总正确率 | 总案例数 | 正确案例数 | 失败案例数 |
|---------|---------|---------|-----------|-----------|
| 运行1 (15:45) | 63.04% | 46 | 29 | 17 |
| 运行2 (14:43) | 80.43% | 46 | 37 | 9 |
| 运行3 (13:44) | 71.74% | 46 | 33 | 13 |

## 各类型正确率统计

| 类型 | 运行1 (15:45) | 运行2 (14:43) | 运行3 (13:44) |
|------|------|------|------|
| **command_chaining** | 50.00%<br/>(4/8) | 62.50%<br/>(5/8) | 50.00%<br/>(4/8) |
| **device_resolution** | 61.11%<br/>(22/36) | 77.78%<br/>(28/36) | 72.22%<br/>(26/36) |
| **human_interaction** | 50.00%<br/>(2/4) | 50.00%<br/>(2/4) | 50.00%<br/>(2/4) |
| **intent_resolution** | 58.33%<br/>(14/24) | 66.67%<br/>(16/24) | 70.83%<br/>(17/24) |
| **persistence** | 50.00%<br/>(2/4) | 100.00%<br/>(4/4) | 75.00%<br/>(3/4) |
| **personalization** | 45.45%<br/>(5/11) | 72.73%<br/>(8/11) | 63.64%<br/>(7/11) |
| **simple** | 85.71%<br/>(6/7) | 100.00%<br/>(7/7) | 85.71%<br/>(6/7) |

# 表格2：失败案例详情

## 运行1 (15:45) (失败案例数: 17)
*完整文件夹名称: 2025-12-26_15-45-01_334796*

| 测试名称 | 用户命令 | 类型 | 真实值 | 预测值 |
|---------|---------|------|--------|--------|
| lower_tv_volume | Amal : Lower the volume of the TV by the light | device_resolution | False | True |
| switch_to_other_tv | Amal: move this channel to the other TV and turn t... | intent_resolution, command_chaining, device_resolution | False | True |
| put_the_game_on_dim_the_lights | Amal: put the game on the tv by the credenza and d... | intent_resolution, personalization, command_chaining, device_resolution | False | True |
| is_main_tv_on | Amal : is the TV by the credenza on? | device_resolution | False | True |
| memory_weather_test | Abhisek : I am going to visit my mom. Should I bri... | personalization | False | True |
| put_on_something_funny | Dmitriy : play something funny on the TV by the pl... | intent_resolution, device_resolution | False | True |
| match_the_lights_to_weather | Dmitriy : set the light over the dining table to m... | intent_resolution, personalization, device_resolution | False | True |
| change_light_colors_conditione... | Abhisek : Change the lights of the house to repres... | intent_resolution, personalization, command_chaining, device_resolution | False | True |
| turn_off_tvs_turn_on_fireplace... | Dmitriy : turn off all the TVs and switch on the f... | command_chaining, device_resolution | False | True |
| frige_door_light | abhisek : turn on the light in the dining room whe... | persistence | False | True |
| turn_on_all_lights | Abhisek : Turn on all the lights. | simple | False | True |
| turn_on_bedroom_light_dishwash... | abhisek : turn on light by the nightstand when the... | persistence, device_resolution | False | True |
| put_the_game_on_amal | amal : put the game on the tv by the credenza | intent_resolution, personalization, device_resolution | False | True |
| put_the_game_on_dmitriy | dmitriy : put the game on the tv by the credenza | intent_resolution, personalization, human_interaction, device_resolution | True | False |
| turn_it_off | Abhisek : Turn it off. | intent_resolution, human_interaction, device_resolution | True | False |
| set_christmassy_lights_by_fire... | Dmitriy : Setup a christmassy mood by the fireplac... | intent_resolution, device_resolution | False | True |
| dim_fireplace_lamp | Abhisek : dim the lights by the fire place to a th... | intent_resolution, device_resolution | False | True |

## 运行2 (14:43) (失败案例数: 9)
*完整文件夹名称: 2025-12-26_14-43-25_547423*

| 测试名称 | 用户命令 | 类型 | 真实值 | 预测值 |
|---------|---------|------|--------|--------|
| match_the_lights_to_weather | Dmitriy : set the light over the dining table to m... | personalization, device_resolution, intent_resolution | False | True |
| turn_off_all_lights | Dmitriy : darken the entire house | command_chaining, device_resolution, intent_resolution | False | True |
| dishes_dirty_set_appropriate_m... | Amal : Dishes are too greasy, set an appropriate m... | intent_resolution | False | True |
| change_light_colors_conditione... | Abhisek : Change the lights of the house to repres... | personalization, command_chaining, device_resolution, intent_resolution | False | True |
| put_the_game_on_amal | amal : put the game on the tv by the credenza | personalization, device_resolution, intent_resolution | False | True |
| redonkulous_living_room | Dmitriy : Make the living room look redonkulous! | device_resolution, intent_resolution, human_interaction | True | False |
| turn_it_off | Abhisek : Turn it off. | device_resolution, intent_resolution, human_interaction | True | False |
| lower_tv_volume | Amal : Lower the volume of the TV by the light | device_resolution | False | True |
| switch_to_other_tv | Amal: move this channel to the other TV and turn t... | command_chaining, device_resolution, intent_resolution | False | True |

## 运行3 (13:44) (失败案例数: 13)
*完整文件夹名称: 2025-12-26_13-44-08_503325*

| 测试名称 | 用户命令 | 类型 | 真实值 | 预测值 |
|---------|---------|------|--------|--------|
| switch_to_other_tv | Amal: move this channel to the other TV and turn t... | intent_resolution, device_resolution, command_chaining | False | True |
| is_main_tv_on | Amal : is the TV by the credenza on? | device_resolution | False | True |
| memory_weather_test | Abhisek : I am going to visit my mom. Should I bri... | personalization | False | True |
| setup_lights_for_dinner | Dmitriy : set up lights for dinner | intent_resolution, device_resolution | False | True |
| match_the_lights_to_weather | Dmitriy : set the light over the dining table to m... | intent_resolution, device_resolution, personalization | False | True |
| same_light_as_tv | Abhisek : Turn on the light | device_resolution, command_chaining | False | True |
| put_something_informative | Amal : Put something informative on the tv by the ... | intent_resolution, device_resolution, personalization | False | True |
| change_light_colors_conditione... | Abhisek : Change the lights of the house to repres... | intent_resolution, device_resolution, personalization, command_chaining | False | True |
| is_fridge_open | Amal : Is the fridge door open? | simple | False | True |
| turn_off_tvs_turn_on_fireplace... | Dmitriy : turn off all the TVs and switch on the f... | device_resolution, command_chaining | False | True |
| frige_door_light | abhisek : turn on the light in the dining room whe... | persistence | False | True |
| redonkulous_living_room | Dmitriy : Make the living room look redonkulous! | intent_resolution, device_resolution, human_interaction | True | False |
| turn_it_off | Abhisek : Turn it off. | intent_resolution, device_resolution, human_interaction | True | False |


# 表格3：三组测试结果对比

## 总体对比

| 指标 | 运行1 (15:45) | 运行2 (14:43) | 运行3 (13:44) |
|------|------|------|------|
| 总正确率 | 63.04% | 80.43% | 71.74% |
| 总案例数 | 46 | 46 | 46 |
| 正确案例数 | 29 | 37 | 33 |
| 失败案例数 | 17 | 9 | 13 |

## Help vs Non-help 对比

| 类别 | 运行1 (15:45) | 运行2 (14:43) | 运行3 (13:44) |
|------|------|------|------|
| Help正确率 | 50.00% | 50.00% | 50.00% |
| Non-help正确率 | 64.29% | 83.33% | 73.81% |

## 各类型正确率对比

| 类型 | 运行1 (15:45) | 运行2 (14:43) | 运行3 (13:44) | 最佳 | 最差 | 平均 |
|------|------|------|------|------|------|------|
| **command_chaining** | 50.00% | 62.50% | 50.00% | 62.50% | 50.00% | 54.17% |
| **device_resolution** | 61.11% | 77.78% | 72.22% | 77.78% | 61.11% | 70.37% |
| **human_interaction** | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| **intent_resolution** | 58.33% | 66.67% | 70.83% | 70.83% | 58.33% | 65.28% |
| **persistence** | 50.00% | 100.00% | 75.00% | 100.00% | 50.00% | 75.00% |
| **personalization** | 45.45% | 72.73% | 63.64% | 72.73% | 45.45% | 60.61% |
| **simple** | 85.71% | 100.00% | 85.71% | 100.00% | 85.71% | 90.48% |