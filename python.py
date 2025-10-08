import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Khởi tạo Gemini Client và API Key (Đã được điều chỉnh để dùng chung) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    GLOBAL_CLIENT = genai.Client(api_key=API_KEY)
except KeyError:
    # Nếu không có API Key, hiển thị cảnh báo và không khởi tạo GLOBAL_CLIENT
    st.error("Lỗi: Không tìm thấy 'GEMINI_API_KEY' trong Streamlit Secrets. Các chức năng AI sẽ không hoạt động.")
    GLOBAL_CLIENT = None
except Exception as e:
    st.error(f"Lỗi khởi tạo Gemini Client: {e}")
    GLOBAL_CLIENT = None

# Đặt tên model mặc định cho chat
CHAT_MODEL_NAME = "gemini-2.5-flash"

# Khởi tạo lịch sử chat trong Session State nếu chưa có
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Hàm tạo/lấy phiên chat đã cache ---
@st.cache_resource
def get_chat_session(_client):
    """Tạo hoặc lấy phiên chat có ghi nhớ lịch sử."""
    if _client is None:
        return None
        
    config = genai.types.GenerateContentConfig(
        system_instruction="Bạn là một trợ lý phân tích tài chính chuyên nghiệp. Hãy trả lời các câu hỏi về các chỉ số tài chính, phương pháp phân tích, hoặc các khái niệm kế toán một cách chính xác và ngắn gọn. Giữ lịch sự và chuyên nghiệp."
    )
    
    try:
        chat = _client.chats.create(
            model=CHAT_MODEL_NAME,
            config=config,
        )
        return chat
    except Exception as e:
        st.error(f"Không thể khởi tạo phiên Chat Gemini: {e}")
        return None

# Lấy phiên chat đã được cache
CHAT_SESSION = get_chat_session(GLOBAL_CLIENT)

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    # Giữ nguyên logic của người dùng
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Giữ nguyên logic cũ) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        # Sử dụng GLOBAL_CLIENT đã khởi tạo ở trên
        client = GLOBAL_CLIENT
        if client is None:
             return "Lỗi: Gemini Client chưa được khởi tạo do thiếu API Key."
             
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng Chính (Main Content Area) ---

uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định để tránh lỗi khi thiếu dữ liệu
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"

            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý lỗi chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A"
                    )
                with col2:
                    delta_value = None
                    if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float):
                         delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                         
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A",
                        delta=delta_value
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                st.warning("Lỗi chia cho 0: Nợ ngắn hạn bằng 0. Không thể tính Chỉ số Thanh toán Hiện hành.")
            
            # --- Chức năng 5: Nhận xét AI (Giữ nguyên) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            # Cần đảm bảo các giá trị N/A được xử lý trước khi gửi
            tt_n_1_display = thanh_toan_hien_hanh_N_1 if isinstance(thanh_toan_hien_hanh_N_1, (float, int)) else "Không có dữ liệu"
            tt_n_display = thanh_toan_hien_hanh_N if isinstance(thanh_toan_hien_hanh_N, (float, int)) else "Không có dữ liệu"

            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{tt_n_1_display}", 
                    f"{tt_n_display}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                if GLOBAL_CLIENT:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, API_KEY)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Chức năng AI Phân tích không khả dụng do thiếu Khóa API.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")


# --- KHUNG CHAT MỚI TRONG SIDEBAR ---
with st.sidebar:
    st.header("💬 Trợ lý Chat Tài chính Gemini")
    st.caption("Hãy hỏi về các khái niệm phân tích hoặc các chỉ số bạn quan tâm.")

    if CHAT_SESSION is None:
        st.warning("Chat không khả dụng. Vui lòng cung cấp `GEMINI_API_KEY` trong Streamlit Secrets.")
    else:
        # 1. Hiển thị lịch sử chat
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 2. Xử lý đầu vào từ người dùng
        if prompt := st.chat_input("Đặt câu hỏi cho Trợ lý...", key="sidebar_chat"):
            
            # Thêm tin nhắn của người dùng vào lịch sử và hiển thị
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gửi tin nhắn đến Gemini
            try:
                with st.spinner('Đang phản hồi...'):
                    # Sử dụng CHAT_SESSION đã được cache để duy trì lịch sử
                    response = CHAT_SESSION.send_message(prompt)
                    full_response = response.text
                
                # Hiển thị phản hồi của AI
                with st.chat_message("assistant"):
                    st.markdown(full_response)
                
                # Thêm phản hồi của AI vào lịch sử
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_message = f"Lỗi: Không thể nhận phản hồi từ Gemini. Chi tiết: {e}"
                with st.chat_message("assistant"):
                    st.error(error_message)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_message})

# -----------------------------------
