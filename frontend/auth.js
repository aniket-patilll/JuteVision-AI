import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';
import { SUPABASE_CONFIG } from './config.js';

export const supabase = createClient(SUPABASE_CONFIG.URL, SUPABASE_CONFIG.KEY);

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
            redirectTo: `${window.location.origin}/dashboard.html`, // Redirect directly to dashboard
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
        window.location.href = 'dashboard.html';
    }
}
