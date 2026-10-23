import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { CasesProvider } from './context/CasesContext';
import { Layout } from './components/Layout';
import { RequireAuth } from './components/RequireAuth';
import { SignIn } from './pages/SignIn';
import { Cases } from './pages/Cases';
import { CaseDetail } from './pages/CaseDetail';
import { AddPatientRecord } from './pages/AddPatientRecord';
import { UploadEmbryoImages } from './pages/UploadEmbryoImages';
import { AuditLog } from './pages/AuditLog';

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
            <Route index element={<Navigate to="/cases" replace />} />
            <Route path="cases" element={<Cases />} />
            <Route path="cases/:id" element={<CaseDetail />} />
            <Route path="add-patient" element={<AddPatientRecord />} />
            <Route path="upload" element={<UploadEmbryoImages />} />
            <Route
              path="audit"
              element={(
                <RequireAuth role="admin">
                  <AuditLog />
                </RequireAuth>
              )}
            />
          </Route>
          <Route path="*" element={<Navigate to="/cases" replace />} />
        </Routes>
      </CasesProvider>
    </AuthProvider>
  );
}
