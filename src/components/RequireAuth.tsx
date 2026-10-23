import { Navigate, useLocation } from 'react-router-dom';
import { useAuth, type Role } from '../context/AuthContext';

export function RequireAuth({ children, role }: { children: React.ReactNode; role?: Role }) {
  const { auth } = useAuth();
  const location = useLocation();

  if (!auth.signedIn) {
    return <Navigate to="/signin" state={{ from: location }} replace />;
  }

  if (role && auth.role !== role) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}
