import React, { useState, useRef } from 'react';
import { Upload, Leaf, AlertCircle, CheckCircle, ChevronRight, Activity, Loader2 } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const diseaseInfo = {
  blight: {
    name: "Bệnh Đốm Lá Lớn (Northern Leaf Blight)",
    desc: "Bệnh do nấm gây ra, tạo ra các vết bệnh hình điếu xì gà trên lá. Nó có thể gây giảm năng suất nghiêm trọng nếu phát triển trước hoặc trong thời kỳ trổ cờ.",
    color: "text-orange-600",
    bg: "bg-orange-100"
  },
  common_rust: {
    name: "Bệnh Gỉ Sắt (Common Rust)",
    desc: "Đặc trưng bởi các mụn mủ màu gỉ sắt trên cả hai mặt lá. Thường ít gây hại hơn các bệnh khác nhưng lây lan nhanh trong điều kiện độ ẩm cao.",
    color: "text-red-600",
    bg: "bg-red-100"
  },
  gray_spot: {
    name: "Bệnh Đốm Xám (Gray Leaf Spot)",
    desc: "Hình thành các vết bệnh hình chữ nhật, màu nâu nhạt đến xám. Phát triển mạnh ở độ ẩm cao và ảnh hưởng nghiêm trọng đến quá trình quang hợp.",
    color: "text-yellow-600",
    bg: "bg-yellow-100"
  },
  healthy: {
    name: "Lá Khỏe Mạnh",
    desc: "Lá có vẻ ngoài khỏe mạnh, không có dấu hiệu nhiễm bệnh. Hãy tiếp tục duy trì các biện pháp chăm sóc nông nghiệp tốt.",
    color: "text-pastel-blue-600",
    bg: "bg-pastel-blue-100"
  },
  not_corn_leaf: {
    name: "Không Phải Lá Bắp",
    desc: "Hình ảnh tải lên dường như không phải là lá bắp (ngô). Vui lòng tải lên một hình ảnh rõ nét của lá bắp để hệ thống phân tích chính xác.",
    color: "text-gray-600",
    bg: "bg-gray-100"
  }
};

function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Vui lòng tải lên một tệp tin hình ảnh.');
        return;
      }
      setSelectedFile(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Phân tích thất bại. Vui lòng kiểm tra lại Backend đã chạy chưa.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans text-gray-800">
      {/* Header */}
      <header className="bg-pastel-blue-700 text-white shadow-lg sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-white p-2 rounded-xl">
              <Leaf className="w-6 h-6 text-pastel-blue-600" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">KimLonVision</h1>
              <p className="text-pastel-blue-100 text-xs font-medium">Hệ thống phân tích & chẩn đoán bệnh lá bắp</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto px-4 py-8 w-full grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Column: Upload */}
        <div className="flex flex-col gap-6">
          <div className="glass-card p-6 flex flex-col items-center">
            <h2 className="text-xl font-semibold mb-4 w-full flex items-center gap-2">
              <Upload className="w-5 h-5 text-pastel-blue-600" /> Tải ảnh lên
            </h2>
            
            <div 
              className={`w-full border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300
                ${imagePreview ? 'border-pastel-blue-400 bg-pastel-blue-50/50' : 'border-gray-300 hover:border-pastel-blue-500 hover:bg-gray-50'}
              `}
              onClick={handleUploadClick}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleImageChange} 
                accept="image/*" 
                className="hidden" 
              />
              
              {imagePreview ? (
                <div className="relative w-full aspect-video rounded-xl overflow-hidden shadow-sm">
                  <img src={imagePreview} alt="Preview" className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                    <p className="text-white font-medium">Nhấp để đổi ảnh khác</p>
                  </div>
                </div>
              ) : (
                <div className="py-12 flex flex-col items-center">
                  <div className="bg-pastel-blue-100 p-4 rounded-full mb-4">
                    <Upload className="w-8 h-8 text-pastel-blue-600" />
                  </div>
                  <p className="text-lg font-medium text-gray-700">Kéo thả hình ảnh vào đây</p>
                  <p className="text-sm text-gray-500 mt-2">hoặc nhấn vào để chọn file từ máy</p>
                  <p className="text-xs text-gray-400 mt-4">Hỗ trợ các định dạng: JPG, JPEG, PNG, WEBP</p>
                </div>
              )}
            </div>

            {error && (
              <div className="mt-4 w-full bg-red-50 text-red-600 p-3 rounded-lg flex items-center gap-2 text-sm font-medium">
                <AlertCircle className="w-4 h-4" /> {error}
              </div>
            )}

            <button 
              onClick={analyzeImage}
              disabled={!selectedFile || loading}
              className={`mt-6 w-full py-4 rounded-xl font-bold text-white text-lg flex items-center justify-center gap-2 transition-all shadow-md
                ${(!selectedFile || loading) ? 'bg-gray-400 cursor-not-allowed' : 'bg-pastel-blue-600 hover:bg-pastel-blue-700 hover:shadow-lg active:scale-[0.98]'}
              `}
            >
              {loading ? (
                <><Loader2 className="w-6 h-6 animate-spin" /> Đang phân tích AI...</>
              ) : (
                <><Activity className="w-6 h-6" /> Chạy Phân Tích</>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="flex flex-col gap-6">
          {result ? (
            <div className="glass-card p-6 animate-in slide-in-from-right-8 duration-500 fade-in">
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 border-b pb-4">
                <CheckCircle className="w-5 h-5 text-pastel-blue-600" /> Kết Quả Phân Tích
              </h2>

              {/* Primary Prediction */}
              <div className={`p-6 rounded-2xl mb-8 flex flex-col items-center text-center ${diseaseInfo[result.predicted_class].bg}`}>
                <p className="text-sm font-medium text-gray-600 uppercase tracking-wider mb-1">Dự đoán chính xác nhất</p>
                <h3 className={`text-3xl font-bold mb-2 ${diseaseInfo[result.predicted_class].color}`}>
                  {diseaseInfo[result.predicted_class].name}
                </h3>
                <div className="inline-flex items-center gap-1 bg-white px-3 py-1 rounded-full shadow-sm text-sm font-bold text-gray-700">
                  Độ tin cậy: {(result.confidence * 100).toFixed(2)}%
                </div>
                <p className="text-gray-700 mt-4 text-sm max-w-md leading-relaxed">
                  {diseaseInfo[result.predicted_class].desc}
                </p>
              </div>

              {/* Probability Distribution */}
              <div>
                <h4 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <ChevronRight className="w-4 h-4" /> Tỷ lệ phân bố các bệnh
                </h4>
                <div className="space-y-4">
                  {Object.entries(result.probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .map(([className, prob]) => (
                    <div key={className} className="flex flex-col gap-1">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium text-gray-700">{diseaseInfo[className].name}</span>
                        <span className="text-gray-500 font-medium">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ease-out ${
                            className === result.predicted_class ? 'bg-pastel-blue-500' : 'bg-gray-300'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-card p-6 flex flex-col items-center justify-center text-center h-full min-h-[400px] border-dashed">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                <Activity className="w-8 h-8 text-gray-300" />
              </div>
              <h3 className="text-xl font-medium text-gray-400 mb-2">Chưa Có Kết Quả</h3>
              <p className="text-gray-400 text-sm max-w-sm">
                Hãy tải lên một hình ảnh lá bắp và nhấp vào "Chạy Phân Tích" để xem kết quả đánh giá của AI.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
