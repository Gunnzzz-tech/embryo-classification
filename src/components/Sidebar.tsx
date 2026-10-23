import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth, type Role } from '../context/AuthContext';
import styles from './Sidebar.module.css';

const navItems: Array<{ to: string; label: string; role?: Role }> = [
  { to: '/cases', label: 'Cases & verification' },
  { to: '/upload', label: 'Upload images' },
  { to: '/add-patient', label: 'Add patient record' },
  { to: '/audit', label: 'Audit log', role: 'admin' },
];

export function Sidebar() {
  const { auth, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = () => {
    signOut();
    navigate('/signin', { replace: true });
  };

  const visibleItems = navItems.filter(
    (item) => !item.role || (auth.signedIn && auth.role === item.role)
  );

  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <span className={styles.name}>Embryo Fertility</span>
        <span className={styles.tagline}>Verified results</span>
      </div>
      <nav className={styles.nav}>
        {visibleItems.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div className={styles.footer}>
        {auth.signedIn && (
          <>
            <p className={styles.userId}>Dr. {auth.doctorId}</p>
            <span className={styles.roleTag}>{auth.role === 'admin' ? 'Admin' : 'Nurse'}</span>
          </>
        )}
        <div className={styles.badge}>ML assessment</div>
        <button type="button" className={styles.signOut} onClick={handleSignOut}>
          Sign out
        </button>
        <p className={styles.copyright}>Admin panel v1.0</p>
      </div>
    </aside>
  );
}
