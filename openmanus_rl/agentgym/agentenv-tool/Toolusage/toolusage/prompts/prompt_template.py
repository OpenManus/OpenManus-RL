prompt_templates = {
    "deepseek":
            """
                {system_prompt}

                USER: {prompt}<｜end▁of▁sentence｜>
                
                ASSISTANT: 
            """,
    "codellama-13b":
            """
                <s>
                <<SYS>>
                {system_prompt}
                <</SYS>>
                [INST]{prompt}[/INST]
            """,
    "codellama-34b":
            """
                <s>[INST]{system_prompt}{prompt}[/INST]
            """,
    "llama":
            """
                <<SYS>>
                {system_prompt}
                <</SYS>>
                [INST]{prompt}[/INST]
            """,
    "lemur":
            """
                <|im_start|>system
                {system_prompt}
                <|im_end|>
                <|im_start|>user
                {prompt}<|im_end|>
                <|im_start|>assistant\n
            """,
    "vicuna":
            """
            {system_prompt}

            USER: {prompt}</s>
            ASSISTANT: 
            """,
    "mistral":
            """
            <s>
            {system_prompt}
            </s>
            [INST]{prompt}[/INST]
            """,
}