import { DocumentView } from "@/components/DocumentView";

export default async function DocumentPage({ params }: PageProps<"/documents/[id]">) {
  const { id } = await params;
  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <DocumentView documentId={id} />
    </main>
  );
}
