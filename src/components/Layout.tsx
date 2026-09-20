import { Outlet, Link, useLocation } from 'react-router-dom';
import { Shield, Home, Search, Clock, Info } from 'lucide-react';

export function Layout() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/analyze', label: 'Analyze', icon: Search },
    { path: '/history', label: 'History', icon: Clock },
    { path: '/about', label: 'About', icon: Info },
  ];
  
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center gap-2">
              <Shield className="w-8 h-8 text-emerald-400" />
              <div>
                <h1 className="text-lg font-bold text-white leading-tight">ForensicAI</h1>
                <p className="text-xs text-gray-400 leading-tight">Image Tampering Detection</p>
              </div>
            </Link>
            
            <nav className="flex items-center gap-1">
              {navItems.map(item => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
      
      {/* Footer */}
      <footer className="border-t border-gray-800 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-sm text-gray-500">
              Image Tampering Detection & Restoration System — AI Forensic Analysis
            </p>
            <p className="text-xs text-gray-600">
              Model: ManTraNet-style CNN | Restoration: Inpainting Reconstruction
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
