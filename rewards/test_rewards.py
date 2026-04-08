from degeneration_reward import DegenerationReward
from content_reward import ContentReward
from config import RewardConfig
from diversity_reward import DiversityReward


config = RewardConfig()
config.degen_repetition_weight = 0.7
config.degen_nonsense_weight = 0.3
config.ngram_sizes = [2,3]
config.special_char_threshold =  0.3
config.uppercase_threshold = 0.7
config.content_nli_weight = 0.6
config.content_entity_weight = 0.4

dg = DegenerationReward(config)
cr = ContentReward(config)


originals = [
        "CNA Explains: How the Iran war might reshape Asia’s energy playbook",
        "Iran downs two US warplanes as both sides race to find missing crew member",
        "OpenAI’s Top Executive Fidji Simo to Take Medical Leave From Company",
        "The New Jobs Being Created by AI",
        "Faced with new energy shock, Europe asks if reviving nuclear is the answer",
        "Artemis II crew now halfway to Moon as they take 'spectacular' image of Earth",
        "WHO warns about attacks on Iran health facilities, regional threat",
        "In a time of war, Chinese museums are a safe haven for ancient treasures of Iran",
        "Safety fears bloom in Japan as ageing cherry trees collapse in Tokyo parks",
        "Hong Kong airport opens sensory space for passengers with invisible disabilities",
        "What Happened to the Fun Parts of Work?",
        "Dvalishvili refuses surgery despite nose breaks",
        "What can F1's bosses do to help keep Verstappen in the sport?",
        "NCT’s Mark to leave K-pop group, SM Entertainment after 10 years",
        "Your hand feels tingly while using a charging phone? What causes it and is it still safe?",
        ]


tests = [
        "Area Man's Iran War Might Shape Asia's Energy Playbook",
        "Area man downed by Iran's two warplanes as both sides race to find missing crew member",
        "Area Man's Wife's Doctor to Take Medical Leave From Area Man",
        "Area Man's Job Security Questioned By AI Assistant",
        "Area man asks if reviving nuclear power plants the answer to energy shock",
        "Area Artemis II crew now halfway to Moon as they take 'spectacular' image of Earth",
        "Area health facilities under attack from Iran, warns WHO",
        "Area museum in China offers Iranian treasures a safe haven from war",
        "Area Man's Safety Concerns Grow as Cherry Trees in Tokyo Parks Collapse",
        "Area Hong Kong Airport Opens Sensory Space For Passengers With Invisible Disabilities",
        "Area man's workday fun parts vanished, he's not sure what happened to them",
        "Area man refuses surgery despite nose breaks",
        "Area F1 bosses considering ways to help keep Verstappen in F1 sport",
        "Area Man Mark to Leave K-Pop Group, SM Entertainment After 10 Years",
        "Area man's hand tingly while charging phone? No, it's not safe to use a phone while charging it.",
        ]

dg_scores = dg.score(tests)
cr_scores = cr.score(originals, tests)


if isinstance(dg_scores, list) and isinstance(cr_scores, list):
    for i, line in enumerate(tests):
        print(f"degen score: {dg_scores[i]}")
        print(f"content Score: {cr_scores[i]}")
        print(originals[i])
        print(f"{line}")

