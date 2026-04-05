-- ============================================================
-- TutorAI Database Schema
-- Run this in your Supabase SQL Editor
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TABLE: profiles
-- ============================================================
CREATE TABLE public.profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username    TEXT UNIQUE,
    full_name   TEXT,
    avatar_url  TEXT,
    grade       TEXT CHECK (grade IN ('11', '12', 'Dropper')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    INSERT INTO public.profiles (id, full_name, avatar_url)
    VALUES (NEW.id, NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'avatar_url');
    RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- ============================================================
-- TABLE: conversations
-- ============================================================
CREATE TABLE public.conversations (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New Conversation',
    subject     TEXT CHECK (subject IN ('physics', 'chemistry', 'mathematics', 'general')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_conversations_user_id ON public.conversations(user_id);

-- ============================================================
-- TABLE: messages
-- ============================================================
CREATE TABLE public.messages (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id  UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    user_id          UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT,
    image_url        TEXT,
    image_ocr_text   TEXT,
    topic_tags       TEXT[],
    is_correct       BOOLEAN,
    token_count      INT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation_id ON public.messages(conversation_id);
CREATE INDEX idx_messages_user_id         ON public.messages(user_id);

-- ============================================================
-- TABLE: weakness_snapshots
-- ============================================================
CREATE TABLE public.weakness_snapshots (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    subject      TEXT NOT NULL CHECK (subject IN ('physics', 'chemistry', 'mathematics')),
    weak_topics  JSONB NOT NULL DEFAULT '[]',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_weakness_user_id ON public.weakness_snapshots(user_id);

-- ============================================================
-- TABLE: quiz_attempts
-- ============================================================
CREATE TABLE public.quiz_attempts (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    subject       TEXT NOT NULL CHECK (subject IN ('physics', 'chemistry', 'mathematics')),
    weak_topics   TEXT[] NOT NULL,
    questions     JSONB NOT NULL,
    responses     JSONB DEFAULT '[]',
    score         INT,
    total         INT NOT NULL DEFAULT 5,
    completed_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_quiz_user_id ON public.quiz_attempts(user_id);

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================
ALTER TABLE public.profiles          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weakness_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_attempts      ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"   ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can CRUD own conversations"
    ON public.conversations FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own messages"
    ON public.messages FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own messages"
    ON public.messages FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own weakness snapshots"
    ON public.weakness_snapshots FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can CRUD own quiz attempts"
    ON public.quiz_attempts FOR ALL USING (auth.uid() = user_id);

-- ============================================================
-- STORAGE BUCKET (run separately or via Dashboard)
-- ============================================================
-- INSERT INTO storage.buckets (id, name, public) VALUES ('question-images', 'question-images', false);
