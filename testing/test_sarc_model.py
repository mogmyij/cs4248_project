from transformers import pipeline
from trl import GRPOTrainer, GRPOConfig

from transformers import pipeline

humour_pipe = pipeline("text-classification", model="mohameddhiab/humor-no-humor")

# Load once at module level (avoid reloading on every call)
sarcasm_pipe = pipeline(
    "text-classification",
    model="surrey-nlp/bertweet-base-finetuned-SARC-combined-DS",
    device=0,          # GPU; use -1 for CPU
    truncation=True,
    max_length=128,
)

def sarcasm_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward function for GRPO.
    Returns high reward if the completion is sarcastic (or non-sarcastic — your choice).
    """
    rewards = []
    for completion in completions:
        result = sarcasm_pipe(completion)[0]  # {'label': 'SARCASM', 'score': 0.92}
        
        # Option A: reward sarcastic outputs
        if result["label"] == "SARCASM":
            rewards.append(result["score"])        # ~0.5–1.0
        else:
            rewards.append(-(1 - result["score"])) # negative reward
            
    return result

def humour_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward function for GRPO.
    Returns high reward if the completion is sarcastic (or non-sarcastic — your choice).
    """
    rewards = []
    for completion in completions:
        result = humour_pipe(completion)[0]  # {'label': 'SARCASM', 'score': 0.92}
        
        # Option A: reward sarcastic outputs
        if result["label"] == "SARCASM":
            rewards.append(result["score"])        # ~0.5–1.0
        else:
            rewards.append(-(1 - result["score"])) # negative reward
            
    return result

shit_lines = ["cant ask the TA's to open? OH ALL I HAD TO DO WAS ASK MY TA",
        "nation now wondering exactly why single finger tingly sensation occurring while holding charging phone",
              
              "nation now knows how iran war might affect future of asian energy playbook",
              "nation now knows iran has downed two u.s. warplanes, both of which have iranian crew members missing",
              "nation informed openai’s fidji simo will be taking medical leave",
              "nation now experiencing creation of new jobs by ai",
            ]

meh_lines = ["nuclear power plant given final chance to save europe from energy crisis",
              "artemis ii crew still 50% of way to moon, but already looking out at earth like it's the coolest thing ever",
              "dvalishvili refuses surgery to fix nose after breaking it again",
              "F1 executives unsure how to prevent driver from simply retiring from sport",
              "Local Man's Hand Still Feeling Normal After Charging Phone For 30 Minutes",
            ]

for l in shit_lines:
    print(l)
    print(sarcasm_reward([],[l]))
    print(humour_reward([],[l]))

print("next")

for l in meh_lines:
    print(l)
    print(sarcasm_reward([],[l]))
    print(humour_reward([],[l]))
