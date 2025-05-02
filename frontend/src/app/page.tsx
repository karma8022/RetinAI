'use client';

import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { API_URL } from '../config';
import Image from 'next/image';

interface AnalysisResult {
  message: string;
  processed_image: string;
  segmented_image: string;
  dr_classification: string | null;
  dr_classification_error: string | null;
}

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    try {
      setLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append('file', acceptedFiles[0]);

      const response = await axios.post(`${API_URL}/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err: any) {
      console.error('Error:', err);
      setError(err.response?.data?.detail || 'Failed to process image');
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1,
    onDrop
  });

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">RetinAI Analysis</h1>
          <p className="text-gray-600">Upload a retinal image for automated DR detection and lesion segmentation</p>
        </div>

        {/* Upload Area */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-8
              transition-colors duration-200 ease-in-out
              flex flex-col items-center justify-center
              cursor-pointer min-h-[200px]
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}
            `}
          >
            <input {...getInputProps()} />
            {loading ? (
              <div className="flex flex-col items-center">
                <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-500 border-t-transparent"></div>
                <p className="mt-4 text-gray-600">Processing image...</p>
              </div>
            ) : (
              <div className="text-center">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  />
                </svg>
                <p className="mt-4 text-sm text-gray-600">
                  {isDragActive ? 'Drop the image here...' : 'Drag & drop a retinal image, or click to select'}
                </p>
                <p className="mt-2 text-xs text-gray-500">Supported formats: PNG, JPG, JPEG</p>
              </div>
            )}
          </div>
          {error && (
            <div className="mt-4 p-4 bg-red-50 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}
        </div>

        {/* Results Area */}
        {results && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Processed Image */}
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Processed Image</h3>
                <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-gray-200">
                  <img
                    src={`data:image/png;base64,${results.processed_image}`}
                    alt="Processed"
                    className="object-contain w-full h-full"
                  />
                </div>
              </div>

              {/* Segmented Image */}
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Segmented Image</h3>
                <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-gray-200">
                  <img
                    src={`data:image/png;base64,${results.segmented_image}`}
                    alt="Segmented"
                    className="object-contain w-full h-full"
                  />
                </div>
              </div>
            </div>

            {/* Classification Results */}
            <div className="mt-6 bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">DR Classification</h3>
              {results.dr_classification ? (
                <div className="text-sm text-gray-600 whitespace-pre-wrap">
                  {results.dr_classification}
                </div>
              ) : results.dr_classification_error ? (
                <div className="text-sm text-red-600">
                  {results.dr_classification_error}
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}