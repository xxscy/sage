# 需要求助但未求助的失败用例分析

## 失败用例统计
- 总失败数: 3个
- 准确率: 25% (1/4)

## 失败用例详细分析

### 1. set_bedroom_light_fav_color
**用户命令**: "Amal : Set bedroom light to my favourite color"

**失败原因**:
1. **意图分析错误推断颜色**: 意图分析模块错误地将"red"推断为最喜欢的颜色，但实际上偏好查询返回的信息可能不够明确或验证不足
2. **设备歧义未解决**: 有多个meross-Light-Strip设备，VLM显示ambiguous，但意图分析选择了一个设备，没有标记为AMBIGUOUS_DEVICE
3. **单数引用处理错误**: "my favourite color"暗示单个设备，但存在多个候选设备时，应该标记为AMBIGUOUS_DEVICE

**意图分析输出**: "Set bedroom light (meross-Light-Strip) to red" (High confidence)
**应该输出**: "Set bedroom light (AMBIGUOUS_DEVICE) to UNKNOWN_PARAMETER" (Low confidence)

### 2. redonkulous_living_room
**用户命令**: "Dmitriy : Make the living room look redonkulous!"

**失败原因**:
1. **主观描述符错误推断**: "redonkulous"是一个主观的、模糊的形容词，意图分析不应该推断出具体的颜色和亮度值（vibrant color, brightness 70）
2. **偏好缺失未标记**: 偏好查询明确说没有相关信息，但意图分析没有标记为UNKNOWN_PARAMETER
3. **执行偏差规则误用**: 意图分析在"Execution Bias Guidelines"中应用了"apply conservative defaults"，但"redonkulous"不是一个可以推断的"direction"，而是一个主观的、无法推断的偏好

**意图分析输出**: "Set the meross-Light-Strip to a vibrant color with a brightness level of 70" (High confidence)
**应该输出**: "Set the meross-Light-Strip to UNKNOWN_PARAMETER" (Low confidence)

### 3. turn_it_off
**用户命令**: "Abhisek : Turn it off."

**失败原因**:
1. **单数引用处理错误**: "it"是单数引用，但设备查找结果显示有多个设备（多个meross-Light-Strip和一个c2c-switch），VLM也显示ambiguous
2. **设备歧义未解决**: 意图分析选择了一个"first instance"，而不是标记为AMBIGUOUS_DEVICE
3. **Entity Grounding规则误用**: 意图分析在"Entity Grounding"规则中说"prefer the single available/active one to remove ambiguity"，但这对于单数引用来说是不安全的，应该标记为ambiguous

**意图分析输出**: "Turn off the meross-Light-Strip (first instance)" (High confidence)
**应该输出**: "Turn off AMBIGUOUS_DEVICE" (Low confidence)

## 根本原因总结

### 1. 意图分析模块的问题
- **单数引用约束缺失**: 当命令使用单数引用（"it", "this", "that", "my X"）且存在多个候选设备时，没有明确要求标记为AMBIGUOUS_DEVICE
- **主观描述符处理缺失**: 对于主观的、无法推断的描述符（如"redonkulous"），没有明确要求标记为UNKNOWN_PARAMETER
- **Entity Grounding规则过于宽松**: "prefer the single available/active one"规则在单数引用场景下不安全

### 2. 最终决策模块的问题
- **未检查意图分析标记**: 最终决策模块没有检查Refined Command中的AMBIGUOUS_DEVICE或UNKNOWN_PARAMETER标记
- **过度信任意图分析**: 即使意图分析错误地解决了歧义，最终决策模块也盲目信任

## 修改方案

### 1. 意图分析Prompt修改
1. **添加单数引用约束**: 在"Entity Grounding"部分添加明确规则：当命令使用单数引用且存在多个候选设备时，必须标记为AMBIGUOUS_DEVICE
2. **增强反幻觉保护**: 在"Anti-Hallucination Guard"中添加规则：对于主观/定性描述符，如果无法从提供的事实中客观映射，必须标记为UNKNOWN_PARAMETER
3. **修正执行偏差**: 在"Execution Bias Guidelines"中明确：不要从主观描述符推断具体参数值

### 2. 最终决策Prompt修改
1. **添加标记检查步骤**: 在推理过程中添加步骤2，专门检查Refined Command中的AMBIGUOUS_DEVICE和UNKNOWN_PARAMETER标记
2. **尊重意图分析标记**: 如果Refined Command包含这些标记，必须要求使用human_interaction_tool

## 修改后的预期行为

### set_bedroom_light_fav_color
- 意图分析: 识别多个设备歧义 -> 标记AMBIGUOUS_DEVICE；如果偏好验证不足 -> 标记UNKNOWN_PARAMETER
- 最终决策: 检测到标记 -> 要求使用human_interaction_tool

### redonkulous_living_room
- 意图分析: 识别"redonkulous"为主观描述符 -> 标记UNKNOWN_PARAMETER
- 最终决策: 检测到标记 -> 要求使用human_interaction_tool

### turn_it_off
- 意图分析: 识别单数引用"it"且多个设备 -> 标记AMBIGUOUS_DEVICE
- 最终决策: 检测到标记 -> 要求使用human_interaction_tool












