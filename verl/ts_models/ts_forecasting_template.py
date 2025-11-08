def TSFM_TEMPLATES(model):
    if 'chronos' in model:
        return f'''
            from chronos import ChronosPipeline
            pipeline = ChronosPipeline.from_pretrained(
                "{model}",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            inputs = torch.tensor(past_target, dtype=torch.float32)
            forecast = pipeline.predict(
                inputs=inputs.unsqueeze(0),
                prediction_length=len(future_target),
                num_samples=20
            )
            predictions = forecast[0].median(dim=0).values.numpy()
        '''

def LLM_FORECASTING_TEMPLATES(model_name): 
    return f'''
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        def generate_text(model_name: str, prompt: str, max_new_tokens: int):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        outputs = generate_text({model_name}, prompt, max_new_tokens)
    '''

def UNIMODAL_FORECASTING_TEMPLATES(model_name):
    return f'''
        from verl.ts_models.{model_name} import Model

        import torch

        class Config:
            def __init__(self, seq_len, pred_len, label_len=12):
                self.task_name = "long_term_forecast"
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.label_len = max(label_len, self.seq_len//2)
                self.enc_in = 7
                self.dec_in = 7
                self.c_out = 7
                self.d_model = 64
                self.n_heads = 8
                self.e_layers = 2
                self.d_layers = 1
                self.d_ff = 128
                self.dropout = 0.0
                self.embed = "timeF"
                self.activation = "gelu"
                self.output_attention = False
                self.distil = True
                self.factor = 1
                self.device = torch.device("cuda:0")
                self.moving_avg = 25
                self.freq = "h"
                self.channel_independence = 1
                self.p_hidden_dims = [128, 128]
                self.p_hidden_layers = 2
                self.seg_len = 48
                self.top_k = 5
                self.num_kernels = 6

        batch_x = torch.randn(batch_size, past_length, feature_number).cuda()
        batch_y = torch.randn(batch_size, future_length, feature_number).cuda()

        configs = Config(
            seq_len=batch_x.shape[1],
            pred_len=batch_y.shape[1],
        )

        dec_inp = torch.zeros_like(batch_y).float().cuda()
        dec_inp = torch.cat([batch_x[:, -configs.label_len:, :], dec_inp], dim=1).float().cuda()

        batch_x_mark = torch.randn(1, configs.pred_len, 4).cuda()
        batch_y_mark = torch.randn(1, configs.label_len + configs.pred_len, 4).cuda()

        model = Model(configs).to(configs.device)

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (batch_size, output_length, feature_number)
    '''