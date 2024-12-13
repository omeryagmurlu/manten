class MoDEAgentProxy:
    def __init__(self, client, use_text_not_embedding):
        self.use_text_not_embedding = use_text_not_embedding
        self.client = client

    def reset(self):
        self.client.reset()

    def step(self, obs, lang_annotation):
        if self.use_text_not_embedding:
            goal = {"lang_text": lang_annotation}
        return self.client.step(obs, goal)
