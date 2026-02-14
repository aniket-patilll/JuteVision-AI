import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';

// Placeholder credentials - YOU MUST UPDATE THESE
const SUPABASE_URL = 'https://nrlogpxkdrflsiukhqcb.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ybG9ncHhrZHJmbHNpdWtocWNiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkxMDk5NTcsImV4cCI6MjA4NDY4NTk1N30.QC6dAmVK6fKainhAs5lltwvDLPJkWg72Wfs4w_nfMRg';

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

export async function signUp(email, password, fullName) {
    const { data, error } = await supabase.auth.signUp({
        email: email,
        password: password,
        options: {
            data: {
                full_name: fullName,
            },
        },
    });
    return { data, error };
}

export async function signIn(email, password) {
    const { data, error } = await supabase.auth.signInWithPassword({
        email: email,
        password: password,
    });
    return { data, error };
}

export async function signInWithGoogle() {
    const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
            redirectTo: window.location.origin, // Redirect back to this page
        }
    });
    return { data, error };
}

export async function signOut() {
    const { error } = await supabase.auth.signOut();
    return { error };
}

export async function resetPassword(email) {
    const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password.html`,
    });
    return { data, error };
}

export async function updatePassword(newPassword) {
    const { data, error } = await supabase.auth.updateUser({
        password: newPassword
    });
    return { data, error };
}

export async function getSession() {
    const { data: { session }, error } = await supabase.auth.getSession();
    return { session, error };
}

// Redirect to login if no session (for protected pages)
export async function requireAuth() {
    const { session } = await getSession();
    if (!session) {
        window.location.href = 'login.html';
    }
    return session;
}

// Redirect to dashboard if already logged in (for login/register pages)
export async function requireNoAuth() {
    const { session } = await getSession();
    if (session) {
        window.location.href = 'index.html';
    }
}
