from rewards.style_reward import StyleReward


sr = StyleReward()


input_strings = [
        "Your hand feels tingly while using a charging phone? What causes it and is it still safe?",
        "NCT's Mark to leave K-pop group, SM Entertainment after 10 years",
        ]
test_strings = [
        "charging phoneilyingly handilyingly tingly sensation possibly safe condition being experienced here possibly not entirely comfortable situation possibly best left alone area of phone being charged not particularly interested in any part of it being touched there least likely to be any other person nearby to notice it anyway little group of friends or family members probably not",
        "sm entertainment-markingly 10-yearly nct markly leavinging grouply situation likely to be well-known situation there any interest in this area least likely place he'll go least likely place he's ever been any of this group's members really interested here least likely place he'll be going any time soon possibly",
        ]

print("testing")

print(sr.score(input_strings, test_strings))

