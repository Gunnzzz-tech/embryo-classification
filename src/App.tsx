import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { CasesProvider } from './context/CasesContext';
import { Layout } from './components/Layout';
import { RequireAuth } from './components/RequireAuth';
import { SignIn } from './pages/SignIn';
import { Dashboard } from './pages/Dashboard';
import { Cases } from './pages/Cases';
import { CaseDetail } from './pages/CaseDetail';
import { AddPatientRecord } from './pages/AddPatientRecord';
import { UploadEmbryoImages } from './pages/UploadEmbryoImages';
import { AuditLog } from './pages/AuditLog';
import { QualityControl } from './pages/QualityControl';

function ProtectedLayout() {
  return (
    <RequireAuth>
      <Layout />
    </RequireAuth>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <CasesProvider>
        <Routes>
          <Route path="/signin" element={<SignIn />} />
          <Route path="/" element={<ProtectedLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="cases" element={<Cases />} />
            <Route path="cases/:id" element={<CaseDetail />} />
            <Route path="add-patient" element={<AddPatientRecord />} />
            <Route path="upload" element={<UploadEmbryoImages />} />
            <Route path="audit" element={<AuditLog />} />
            <Route path="quality" element={<QualityControl />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </CasesProvider>
    </AuthProvider>
  );
}
