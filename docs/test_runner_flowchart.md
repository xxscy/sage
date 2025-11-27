```mermaid
flowchart TB
    CASE_ENTRY["case_func 入口<br/>输入: deepcopy 的 device_state + coordinator_config"] --> SETUP
    SETUP["testing_utils.setup<br/>输出: test_id, coordinator<br/>副作用: fake DB 写入初始状态,\ncondition server reset"] --> USER_CMD
    USER_CMD["case_func 构造自然语言命令<br/>例: Amal: turn on the TV"] --> EXECUTE

    subgraph COORDINATOR["coordinator.execute(user_command)"]
        EXECUTE --> STATE_SUMMARY["SAGE Planner 汇总上下文<br/>输入: 设备状态摘要、用例指示\n输出: 给 LLM 的提示"]
        STATE_SUMMARY --> PLAN_LLM["LLM 生成执行计划<br/>输出样例:<br/>1) 识别 TV by credenza IDs<br/>2) 若多设备 → device_disambiguation<br/>3) 关机 → execute_command"]
        PLAN_LLM --> DEVICE_DIS["device_disambiguation (可选)<br/>输入: [device_ids], 描述=by the credenza<br/>输出: 精确设备 ID"]
        PLAN_LLM --> API_DOC["api_doc_retrieval (可选)<br/>输入: [{device_id, capability}]<br/>输出: 支持的 component/属性"]
        DEVICE_DIS --> TOOL_ROUTER
        API_DOC --> TOOL_ROUTER
        PLAN_LLM --> TOOL_ROUTER

        subgraph TOOL_ROUTER["按计划触发工具"]
            TOOL_ROUTER --> GET_ATTR["get_attribute<br/>输入示例:{device_id:'8e20',component:'main',capability:'switch',attribute:'switch'}<br/>输出: JSON value→用于思考"]
            TOOL_ROUTER --> EXEC_CMD["execute_command<br/>输入示例:{device_id:'8e20',component:'main',capability:'switch',command:'on',args:[]}<br/>输出: SmartThings(faked) ack，设备状态写入 fake DB"]
            TOOL_ROUTER --> TV_SCHED["tv_schedule_search (需要节目单时)<br/>输入:{source:'montreal-fibe-tv',query:'basketball'}<br/>输出: Channel 列表"]
            TOOL_ROUTER --> WEATHER["weather_tool<br/>输入:'Montreal, Canada'<br/>输出: 天气摘要"]
            TOOL_ROUTER --> HUMAN["human_interaction_tool (若需澄清)<br/>输入:{query:'Which TV do you mean?'}<br/>输出: (user_name, response) 或 ('','')"]
        end

        EXEC_CMD --> STATE_CHECK["Executor 读取 fake DB 最新状态\n用于后续步骤"] --> PLAN_LLM
        GET_ATTR --> PLAN_LLM
        TV_SCHED --> PLAN_LLM
        WEATHER --> PLAN_LLM
        HUMAN --> PLAN_LLM

        PLAN_LLM --> FINAL_ANSWER["生成最终回答<br/>输出: 自然语言 + 说明执行的 device IDs"]
    end

    FINAL_ANSWER --> ASSERT["testcases 断言<br/>输入: fake DB 状态或外部 API 结果\n例: assert switch=='on'"] --> RESULT
    RESULT{"通过?"} -->|是| SUCCESS["记录 success, runtime"] --> LOG
    RESULT -->|否| FAILURE["捕获异常, error=traceback"] --> LOG
    LOG["merge_test_types + 写 test_log JSON<br/>human_interaction_stats 一并写入"] --> END["case_func 返回，等待下一个 case"]
```

