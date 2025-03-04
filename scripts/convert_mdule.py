import os
os.chdir("..")

model = CTC_encoder(n_hid=512)
model.load_state_dict(torch.load("/data1/YHC/Model_Save/CTC_epoch:20_loss:0.235789_model.pt"))
model = model.cuda()
model.eval()

datas = torch.rand(32, 1, 6000).cuda().half()
model.half()
traced_script_module = torch.jit.trace(model, datas)
traced_script_module.save("/public3/YHC/model_merge_ara_oryza_fruitfly/CTC_0922_script_epoch:20_loss:0.267006_model.pt")
