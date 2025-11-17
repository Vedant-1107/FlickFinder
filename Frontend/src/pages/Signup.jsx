// src/pages/Signup.jsx
import React, { useState } from "react";
import { User, Mail, Lock, Film, Eye, EyeOff, CheckCircle } from "lucide-react";
import { signup } from "../api";
import { useNavigate } from "react-router-dom";

export default function Signup() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const passwordChecks = {
    length: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
  };

  const isPasswordStrong = Object.values(passwordChecks).every(Boolean);

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    if (!username || !email || !password) {
      alert("Please fill in all fields");
      return;
    }
    if (!isPasswordStrong) {
      alert("Password doesn't meet strength requirements");
      return;
    }
    setLoading(true);
    try {
      await signup({ username, email, password });
      alert("Account created successfully — please login");
      navigate("/login");
    } catch (err) {
      alert(err?.message ?? "Signup failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-600 via-emerald-600 to-teal-700 flex items-center justify-center p-4">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-green-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-emerald-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-40 w-80 h-80 bg-teal-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <div className="w-full max-w-md relative">
        <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-2xl p-8 backdrop-blur-sm bg-opacity-95">
          <div className="flex justify-center mb-8">
            <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-4 rounded-2xl shadow-lg">
              <Film className="w-10 h-10 text-white" />
            </div>
          </div>

          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-2">Create Account</h2>
            <p className="text-gray-600">Join us and discover amazing movies</p>
          </div>

          <div className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Username</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input type="text" required value={username} onChange={(e) => setUsername(e.target.value)} className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all outline-none" placeholder="Choose a username" autoComplete="username" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all outline-none" placeholder="you@example.com" autoComplete="email" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input type={showPassword ? "text" : "password"} required value={password} onChange={(e) => setPassword(e.target.value)} className="w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all outline-none" placeholder="Create a strong password" autoComplete="new-password" />
                <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors">
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>

              {password && (
                <div className="mt-3 space-y-2">
                  <div className="flex items-center gap-2 text-xs">
                    <CheckCircle className={`w-4 h-4 ${passwordChecks.length ? 'text-green-500' : 'text-gray-300'}`} />
                    <span className={passwordChecks.length ? 'text-gray-700' : 'text-gray-500'}>At least 8 characters</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <CheckCircle className={`w-4 h-4 ${passwordChecks.uppercase ? 'text-green-500' : 'text-gray-300'}`} />
                    <span className={passwordChecks.uppercase ? 'text-gray-700' : 'text-gray-500'}>One uppercase letter</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <CheckCircle className={`w-4 h-4 ${passwordChecks.number ? 'text-green-500' : 'text-gray-300'}`} />
                    <span className={passwordChecks.number ? 'text-gray-700' : 'text-gray-500'}>One number</span>
                  </div>
                </div>
              )}
            </div>

            <label className="flex items-start cursor-pointer">
              <input type="checkbox" className="w-4 h-4 mt-0.5 text-green-600 border-gray-300 rounded focus:ring-green-500" />
              <span className="ml-2 text-sm text-gray-600">I agree to the{" "}<button className="text-green-600 hover:text-green-700 font-medium">Terms of Service</button>{" "}and{" "}<button className="text-green-600 hover:text-green-700 font-medium">Privacy Policy</button></span>
            </label>

            <button type="submit" disabled={loading || !isPasswordStrong} className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white py-3 rounded-lg font-semibold hover:from-green-700 hover:to-emerald-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:-translate-y-0.5">
              {loading ? <span className="flex items-center justify-center gap-2"><div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>Creating account...</span> : "Create Account"}
            </button>
          </div>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-white text-gray-500">or sign up with</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <button type="button" onClick={() => alert("Google signup not configured")} className="flex items-center justify-center gap-2 py-2 px-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
              <span className="text-sm font-medium text-gray-700">Google</span>
            </button>
            <button type="button" onClick={() => alert("Facebook signup not configured")} className="flex items-center justify-center gap-2 py-2 px-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
              <span className="text-sm font-medium text-gray-700">Facebook</span>
            </button>
          </div>

          <p className="text-center text-sm text-gray-600 mt-6">
            Already have an account? <a href="/login" className="text-green-600 hover:text-green-700 font-semibold transition-colors">Sign in</a>
          </p>
        </form>

        <p className="text-center text-sm text-white/80 mt-6">Protected by industry-standard encryption</p>
      </div>
    </div>
  );
}