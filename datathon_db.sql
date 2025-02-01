--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 17.2

-- Started on 2025-02-02 00:14:26 +08

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 4 (class 2615 OID 2200)
-- Name: public; Type: SCHEMA; Schema: -; Owner: pg_database_owner
--

CREATE SCHEMA public;


ALTER SCHEMA public OWNER TO pg_database_owner;

--
-- TOC entry 4307 (class 0 OID 0)
-- Dependencies: 4
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: pg_database_owner
--

COMMENT ON SCHEMA public IS 'standard public schema';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 219 (class 1259 OID 16514)
-- Name: article_gpe; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.article_gpe (
    article_id bigint NOT NULL,
    gpe_id bigint NOT NULL
);


ALTER TABLE public.article_gpe OWNER TO postgres;

--
-- TOC entry 215 (class 1259 OID 16459)
-- Name: articles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.articles (
    article_id bigint NOT NULL,
    summary text NOT NULL,
    uuid uuid NOT NULL,
    link text,
    zeroshot_labels text[],
    persons text[],
    orgs text[]
);


ALTER TABLE public.articles OWNER TO postgres;

--
-- TOC entry 216 (class 1259 OID 16464)
-- Name: articles_article_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.articles ALTER COLUMN article_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.articles_article_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 218 (class 1259 OID 16492)
-- Name: gpe; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gpe (
    gpe_id bigint NOT NULL,
    name text
);


ALTER TABLE public.gpe OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 16491)
-- Name: gpe_gpe_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.gpe ALTER COLUMN gpe_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.gpe_gpe_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 4156 (class 2606 OID 16518)
-- Name: article_gpe article_gpe_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.article_gpe
    ADD CONSTRAINT article_gpe_pkey PRIMARY KEY (article_id, gpe_id);


--
-- TOC entry 4150 (class 2606 OID 16466)
-- Name: articles articles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_pkey PRIMARY KEY (article_id);


--
-- TOC entry 4152 (class 2606 OID 16498)
-- Name: gpe gpe_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gpe
    ADD CONSTRAINT gpe_pkey PRIMARY KEY (gpe_id);


--
-- TOC entry 4154 (class 2606 OID 16531)
-- Name: gpe name_unique; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gpe
    ADD CONSTRAINT name_unique UNIQUE (name);


--
-- TOC entry 4157 (class 2606 OID 16519)
-- Name: article_gpe article_gpe_article_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.article_gpe
    ADD CONSTRAINT article_gpe_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(article_id) ON DELETE CASCADE;


--
-- TOC entry 4158 (class 2606 OID 16524)
-- Name: article_gpe article_gpe_gpe_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.article_gpe
    ADD CONSTRAINT article_gpe_gpe_id_fkey FOREIGN KEY (gpe_id) REFERENCES public.gpe(gpe_id) ON DELETE CASCADE;


-- Completed on 2025-02-02 00:14:27 +08

--
-- PostgreSQL database dump complete
--

