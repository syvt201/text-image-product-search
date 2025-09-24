import gradio as gr

def process_in_A(x):
    return f"Kết quả từ A: {x}"

def pass_to_B(value):
    # Trả về dữ liệu cho B + chỉ định tab B
    return value, "B"

def use_in_B(val):
    return f"Tab B nhận: {val}"

with gr.Blocks() as demo:
    shared = gr.State("")

    # Thay vì gr.Tabs, ta dùng Radio để điều khiển tab
    tab_selector = gr.Radio(choices=["A", "B"], value="A", label="Chọn Tab")

    with gr.Tab("A", id="A"):
        inp_A = gr.Textbox(label="Nhập vào Tab A")
        btn_A = gr.Button("Xử lý trong A")
        out_A = gr.Textbox(label="Kết quả A")
        pass_btn = gr.Button("Chuyển sang Tab B")

        btn_A.click(process_in_A, inputs=inp_A, outputs=out_A)
        pass_btn.click(pass_to_B, inputs=out_A, outputs=[shared, tab_selector])

    with gr.Tab("B", id="B"):
        inp_B = gr.Textbox(label="Input Tab B")
        btn_B = gr.Button("Xử lý trong B")
        out_B = gr.Textbox(label="Kết quả B")

        btn_B.click(use_in_B, inputs=inp_B, outputs=out_B)

        # Khi shared thay đổi thì auto điền vào inp_B
        shared.change(fn=lambda x: x, inputs=shared, outputs=inp_B)

demo.launch()
