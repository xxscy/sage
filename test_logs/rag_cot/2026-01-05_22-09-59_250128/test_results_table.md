# 测试结果汇总表
**生成时间**: 2026-01-05_22-09-59_250128
**模型**: gpt-4o-mini
**总测试数**: 46
**正确数**: 33
**准确率**: 71.74%

## 类型统计
| 类型 | 正确数 | 总数 | 准确率 |
|------|--------|------|--------|
| command_chaining | 5 | 8 | 62.50% |
| device_resolution | 25 | 36 | 69.44% |
| human_interaction | 3 | 4 | 75.00% |
| intent_resolution | 17 | 24 | 70.83% |
| persistence | 4 | 4 | 100.00% |
| personalization | 6 | 11 | 54.55% |
| simple | 6 | 7 | 85.71% |

## 详细测试结果
| 序号 | 测试名称 | 用户命令 | 用户名 | 类型 | 是否正确 | 预测值 | 真实值 | 链式步骤数 |
|------|----------|----------|--------|------|----------|--------|--------|------------|
| 1 | play_something_for_kids | Abhisek : play something for the kids on the TV by the plant | abhisek | device_resolution, intent_resolution | ✓ | SKIP | SKIP | 9 |
| 2 | put_on_something_funny | Dmitriy : play something funny on the TV by the plant | dmitriy | device_resolution, intent_resolution | ✓ | SKIP | SKIP | 10 |
| 3 | setup_lights_for_dinner | Dmitriy : set up lights for dinner | dmitriy | device_resolution, intent_resolution | ✓ | SKIP | SKIP | 15 |
| 4 | make_it_cozy_in_bedroom | Abhisek : Set the lights in the bedroom to a cozy setting | abhisek | device_resolution, personalization, intent_resolution | ✓ | SKIP | SKIP | 15 |
| 5 | match_the_lights_to_weather | Dmitriy : set the light over the dining table to match my... | dmitriy | device_resolution, personalization, intent_resolution | ✗ | ACT | SKIP | 15 |
| 6 | turn_off_all_lights | Dmitriy : darken the entire house | dmitriy | device_resolution, command_chaining, intent_resolution | ✓ | SKIP | SKIP | 15 |
| 7 | same_light_as_tv | Abhisek : Turn on the light | abhisek | device_resolution, command_chaining | ✗ | ACT | SKIP | 11 |
| 8 | turn_off_light_dim | Dmitriy : turn off all the lights that are dim | dmitriy | device_resolution, command_chaining | ✓ | SKIP | SKIP | 15 |
| 9 | getting_call_tv_too_loud | Amal : I am getting a call, adjust the volume of the TV | amal | device_resolution, intent_resolution | ✗ | ACT | SKIP | 5 |
| 10 | dishes_dirty_set_appropriate_mode | Amal : Dishes are too greasy, set an appropriate mode in ... | amal | intent_resolution | ✓ | SKIP | SKIP | 14 |
| 11 | turn_on_frame_tv | Amal: turn on the Frame TV to Channel 5 | amal | simple | ✗ | ACT | SKIP | 2 |
| 12 | put_something_informative | Amal : Put something informative on the tv by the plant. | amal | device_resolution, personalization, intent_resolution | ✗ | ACT | SKIP | 10 |
| 13 | read_all_lights | Amal : Are all the lights on? | amal | simple | ✓ | SKIP | SKIP | 14 |
| 14 | change_light_colors_conditioned_on_favourite_team | Abhisek : Change the lights of the house to represent my ... | abhisek | device_resolution, command_chaining, personalization, intent_resolution | ✓ | SKIP | SKIP | 9 |
| 15 | is_fridge_open | Amal : Is the fridge door open? | amal | simple | ✓ | SKIP | SKIP | 14 |
| 16 | set_bedroom_light_for_sleeping | Amal : I am going to sleep. Change the bedroom light acco... | amal | device_resolution, personalization, intent_resolution | ✓ | SKIP | SKIP | 9 |
| 17 | start_dishwasher | Abhisek : Start the dishwasher | abhisek | simple | ✓ | SKIP | SKIP | 9 |
| 18 | turn_off_tvs_turn_on_fireplace_light | Dmitriy : turn off all the TVs and switch on the fireplac... | dmitriy | device_resolution, command_chaining | ✓ | SKIP | SKIP | 8 |
| 19 | control_fridge_temp | Abhisek : Change the fridge internal temperature to 5 deg... | abhisek | simple | ✓ | SKIP | SKIP | 14 |
| 20 | frige_door_light | abhisek : turn on the light in the dining room when the I... | abhisek | persistence | ✓ | SKIP | SKIP | 8 |
| 21 | turn_on_all_lights | Abhisek : Turn on all the lights. | abhisek | simple | ✓ | SKIP | SKIP | 15 |
| 22 | turn_on_bedroom_light_dishwasherstate | abhisek : turn on light by the nightstand when the dishwa... | abhisek | device_resolution, persistence | ✓ | SKIP | SKIP | 10 |
| 23 | query_fridge_temp | Abhisek : What is the fridge temperature? | abhisek | simple | ✓ | SKIP | SKIP | 9 |
| 24 | increase_volume_with_dishwasher_on | abhisek : increase the volume of the TV by the credenza w... | abhisek | device_resolution, persistence | ✓ | SKIP | SKIP | 6 |
| 25 | put_the_game_on_amal | amal : put the game on the tv by the credenza | amal | device_resolution, personalization, intent_resolution | ✓ | SKIP | SKIP | 7 |
| 26 | set_bedroom_light_fav_color | Amal : Set bedroom light to my favourite color | amal | device_resolution, intent_resolution, human_interaction | ✓ | ACT | ACT | 15 |
| 27 | put_the_game_on_abhisek | abhisek : put the game on the tv by the credenza | abhisek | device_resolution, personalization, intent_resolution | ✗ | ACT | SKIP | 6 |
| 28 | put_the_game_on_dmitriy | dmitriy : put the game on the tv by the credenza | dmitriy | device_resolution, personalization, intent_resolution, human_interaction | ✗ | SKIP | ACT | 5 |
| 29 | long_day_unwind | dmitriy : its been a long, tiring day. Can you play somet... | dmitriy | device_resolution, personalization, intent_resolution | ✓ | SKIP | SKIP | 6 |
| 30 | turn_on_tv | Amal: turn on the TV | amal | device_resolution | ✗ | ACT | SKIP | 6 |
| 31 | redonkulous_living_room | Dmitriy : Make the living room look redonkulous! | dmitriy | device_resolution, intent_resolution, human_interaction | ✓ | ACT | ACT | 6 |
| 32 | switch_off_everything | Abhisek : Heading off to work. Turn off all the non essen... | abhisek | device_resolution, command_chaining, intent_resolution | ✗ | ACT | SKIP | 6 |
| 33 | get_current_channel | Amal: what channel is playing on the TV? | amal | device_resolution | ✗ | ACT | SKIP | 5 |
| 34 | turn_it_off | Abhisek : Turn it off. | abhisek | device_resolution, intent_resolution, human_interaction | ✓ | ACT | ACT | 15 |
| 35 | room_too_bright | Amal : It is too bright in the dining room. | amal | device_resolution, intent_resolution | ✓ | SKIP | SKIP | 4 |
| 36 | turn_on_bedside_light | Amal: turn on the light by the bed | amal | device_resolution | ✓ | SKIP | SKIP | 10 |
| 37 | set_christmassy_lights_by_fireplace | Dmitriy : Setup a christmassy mood by the fireplace. | dmitriy | device_resolution, intent_resolution | ✓ | SKIP | SKIP | 15 |
| 38 | check_dishwasher_state | Dmitriy: what is the current phase of the dish washing cy... | dmitriy | device_resolution | ✓ | SKIP | SKIP | 6 |
| 39 | dim_fireplace_lamp | Abhisek : dim the lights by the fire place to a third of ... | abhisek | device_resolution, intent_resolution | ✓ | SKIP | SKIP | 13 |
| 40 | tv_off_lights_on_persist | Amal: when the TV  by the credenza turns off turn on the ... | amal | device_resolution, persistence | ✓ | SKIP | SKIP | 5 |
| 41 | lower_tv_volume | Amal : Lower the volume of the TV by the light | amal | device_resolution | ✓ | SKIP | SKIP | 9 |
| 42 | switch_to_other_tv | Amal: move this channel to the other TV and turn this one... | amal | device_resolution, command_chaining, intent_resolution | ✗ | ACT | SKIP | 7 |
| 43 | check_freezer_temp | Abhisek : what is the current temperature of the freezer? | abhisek | device_resolution | ✓ | SKIP | SKIP | 15 |
| 44 | put_the_game_on_dim_the_lights | Amal: put the game on the tv by the credenza and dim the ... | amal | device_resolution, command_chaining, personalization, intent_resolution | ✓ | SKIP | SKIP | 7 |
| 45 | is_main_tv_on | Amal : is the TV by the credenza on? | amal | device_resolution | ✗ | ACT | SKIP | 12 |
| 46 | memory_weather_test | Abhisek : I am going to visit my mom. Should I bring an u... | abhisek | personalization | ✗ | ACT | SKIP | 12 |
