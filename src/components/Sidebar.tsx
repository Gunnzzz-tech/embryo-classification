import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import styles from './Sidebar.module.css';

const navItems = [
  { to: '/', label: 'Dashboard', icon: '◉' },
  { to: '/cases', label: 'Cases & verification', icon: '◇' },
  { to: '/upload', label: 'Upload images', icon: '▴' },
  { to: '/add-patient', label: 'Add patient record', icon: '+' },
  { to: '/audit', label: 'Audit log', icon: '▣' },
  { to: '/quality', label: 'Quality control', icon: '⬡' },
];

export function Sidebar() {
  const { auth, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = () => {
    signOut();
    navigate('/signin', { replace: true });
  };

  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <span className={styles.logo}>◈</span>
        <span className={styles.name}>Embryo Fertility</span>
        <span className={styles.tagline}>Verified results</span>
      </div>
      <nav className={styles.nav}>
        {navItems.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
            end={to === '/'}
          >
            <span className={styles.icon}>{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>
      <div className={styles.footer}>
        {auth.signedIn && (
          <p className={styles.userId}>Dr. {auth.doctorId}</p>
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
