#!/usr/bin/env python3
"""
XML Parser for Roca Real Estate Listings
Parses the roca.xml file containing property listings.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class PropertyImage:
    """Represents a property image"""
    url: str


@dataclass
class PropertyPrice:
    """Represents a property price"""
    amount: Decimal
    currency: str
    operation: str  # ALQUILER (rent) or VENTA (sale)


@dataclass
class PropertyCharacteristic:
    """Represents a property characteristic"""
    id: str
    name: str
    value: str = ""
    value_id: str = ""


@dataclass
class PropertyLocation:
    """Represents property location"""
    postal_code: str = ""
    address: str = ""
    locality: str = ""
    latitude: str = ""
    longitude: str = ""
    show_map: str = ""


@dataclass
class Property:
    """Represents a real estate property"""
    code: str
    reference: str
    title: str
    description: str
    property_type: str = ""
    property_subtype: str = ""
    characteristics: List[PropertyCharacteristic] = field(default_factory=list)
    prices: List[PropertyPrice] = field(default_factory=list)
    images: List[PropertyImage] = field(default_factory=list)
    location: PropertyLocation = field(default_factory=PropertyLocation)
    publisher_code: str = ""
    publisher_name: str = ""
    publisher_phone: str = ""


class RocaXMLParser:
    """Parser for Roca real estate XML files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.properties: List[Property] = []
    
    def parse(self) -> List[Property]:
        """Parse the XML file and return list of properties"""
        # Parse XML incrementally to handle large files
        context = ET.iterparse(self.file_path, events=('start', 'end'))
        context = iter(context)
        
        # Get root element
        event, root = next(context)
        
        for event, elem in context:
            if event == 'end' and elem.tag == 'Imovel':
                property_obj = self._parse_property(elem)
                if property_obj:
                    self.properties.append(property_obj)
                # Clear element to free memory
                elem.clear()
                root.clear()
        
        return self.properties
    
    def _get_text(self, element, tag: str, default: str = "") -> str:
        """Safely get text from XML element"""
        child = element.find(tag)
        return child.text if child is not None and child.text else default
    
    def _parse_property(self, imovel_elem) -> Property:
        """Parse a single property (Imovel) element"""
        try:
            # Basic info
            code = self._get_text(imovel_elem, 'codigoAnuncio')
            reference = self._get_text(imovel_elem, 'codigoReferencia')
            title = self._get_text(imovel_elem, 'titulo')
            description = self._get_text(imovel_elem, 'descricao')
            
            property_obj = Property(
                code=code,
                reference=reference,
                title=title,
                description=description
            )
            
            # Property type
            tipo_prop = imovel_elem.find('tipoPropriedade')
            if tipo_prop is not None:
                property_obj.property_type = self._get_text(tipo_prop, 'tipo')
                property_obj.property_subtype = self._get_text(tipo_prop, 'subTipo')
            
            # Characteristics
            caracteristicas = imovel_elem.find('caracteristicas')
            if caracteristicas is not None:
                for carac in caracteristicas.findall('caracteristica'):
                    char = PropertyCharacteristic(
                        id=self._get_text(carac, 'id'),
                        name=self._get_text(carac, 'nome'),
                        value=self._get_text(carac, 'valor'),
                        value_id=self._get_text(carac, 'idValor')
                    )
                    property_obj.characteristics.append(char)
            
            # Prices
            precos = imovel_elem.find('precos')
            if precos is not None:
                for preco in precos.findall('preco'):
                    try:
                        price = PropertyPrice(
                            amount=Decimal(self._get_text(preco, 'quantidade', '0')),
                            currency=self._get_text(preco, 'moeda'),
                            operation=self._get_text(preco, 'operacao')
                        )
                        property_obj.prices.append(price)
                    except Exception:
                        pass
            
            # Images
            multimidia = imovel_elem.find('multimidia')
            if multimidia is not None:
                imagens = multimidia.find('imagens')
                if imagens is not None:
                    for img in imagens.findall('imagem'):
                        url = self._get_text(img, 'urlImagem')
                        if url:
                            property_obj.images.append(PropertyImage(url=url))
            
            # Location
            localizacao = imovel_elem.find('localizacao')
            if localizacao is not None:
                property_obj.location = PropertyLocation(
                    postal_code=self._get_text(localizacao, 'codigoPostal'),
                    address=self._get_text(localizacao, 'endereco'),
                    locality=self._get_text(localizacao, 'localidade'),
                    latitude=self._get_text(localizacao, 'latitude'),
                    longitude=self._get_text(localizacao, 'longitude'),
                    show_map=self._get_text(localizacao, 'mostrarMapa')
                )
            
            # Publisher
            publicador = imovel_elem.find('publicador')
            if publicador is not None:
                property_obj.publisher_code = self._get_text(publicador, 'codigoImobiliaria')
                property_obj.publisher_name = self._get_text(publicador, 'nomeContato')
                property_obj.publisher_phone = self._get_text(publicador, 'telefoneContato')
            
            return property_obj
            
        except Exception as e:
            print(f"Error parsing property: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about parsed properties"""
        stats = {
            'total_properties': len(self.properties),
            'for_sale': 0,
            'for_rent': 0,
            'property_types': {},
            'total_images': 0,
            'properties_with_location': 0,
        }
        
        for prop in self.properties:
            # Count sale vs rent
            for price in prop.prices:
                if price.operation == 'VENTA':
                    stats['for_sale'] += 1
                elif price.operation == 'ALQUILER':
                    stats['for_rent'] += 1
            
            # Count property types
            if prop.property_type:
                stats['property_types'][prop.property_type] = \
                    stats['property_types'].get(prop.property_type, 0) + 1
            
            # Count images
            stats['total_images'] += len(prop.images)
            
            # Count properties with location
            if prop.location and prop.location.address:
                stats['properties_with_location'] += 1
        
        return stats
    
    def search_properties(self, keyword: str = None, 
                         property_type: str = None,
                         operation: str = None) -> List[Property]:
        """Search properties by keyword, type, or operation"""
        results = []
        
        for prop in self.properties:
            # Check keyword in title or description
            if keyword:
                keyword_lower = keyword.lower()
                if (keyword_lower not in prop.title.lower() and 
                    keyword_lower not in prop.description.lower()):
                    continue
            
            # Check property type
            if property_type and prop.property_type != property_type:
                continue
            
            # Check operation (sale/rent)
            if operation:
                if not any(p.operation == operation for p in prop.prices):
                    continue
            
            results.append(prop)
        
        return results


def main():
    """Main function to demonstrate the parser"""
    import sys
    
    # File path
    xml_file = 'roca.xml'
    
    print(f"Parsing {xml_file}...")
    parser = RocaXMLParser(xml_file)
    
    try:
        properties = parser.parse()
        print(f"✓ Successfully parsed {len(properties)} properties\n")
        
        # Show statistics
        stats = parser.get_statistics()
        print("=== Statistics ===")
        print(f"Total properties: {stats['total_properties']}")
        print(f"For sale: {stats['for_sale']}")
        print(f"For rent: {stats['for_rent']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Properties with location: {stats['properties_with_location']}")
        print("\nProperty types:")
        for ptype, count in stats['property_types'].items():
            print(f"  - {ptype}: {count}")
        
        # Show first property as example
        if properties:
            print("\n=== First Property Example ===")
            prop = properties[0]
            print(f"Code: {prop.code}")
            print(f"Title: {prop.title[:80]}...")
            print(f"Type: {prop.property_type} - {prop.property_subtype}")
            print(f"Location: {prop.location.address}")
            if prop.prices:
                price = prop.prices[0]
                print(f"Price: {price.currency} {price.amount:,.2f} ({price.operation})")
            print(f"Images: {len(prop.images)}")
            print(f"Characteristics: {len(prop.characteristics)}")
            
            # Show some characteristics
            if prop.characteristics:
                print("\nKey characteristics:")
                for char in prop.characteristics[:5]:
                    if char.value:
                        print(f"  - {char.name}: {char.value}")
        
        # Example search
        print("\n=== Example Search: 'apartamento' ===")
        results = parser.search_properties(keyword='apartamento')
        print(f"Found {len(results)} apartments")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
